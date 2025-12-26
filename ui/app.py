#!/usr/bin/env python3
"""
MLX Whisper 语音转录 UI

功能：
- 支持 Large-v3 和 Large-v3-turbo 模型
- 预设场景模式（标准、电话录音、会议、嘈杂环境）
- 场景预设联动所有参数
- 可选说话人分离（需要 HuggingFace Token）
- 高级参数始终可见可调
- 实时日志输出（无进度条，直接文字打印）
- 单文件可选转录前 N 分钟
- 下载 TXT/JSON/SRT 功能
"""

import os
import sys
import time
import json
import tempfile
import traceback
from pathlib import Path
from typing import Optional, Tuple, List, Generator
from dataclasses import dataclass

import gradio as gr
import numpy as np
import librosa

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ========== 配置常量 ==========

# 模型配置
MODELS = {
    "Large-v3-turbo (快速)": {
        "name": "large-v3-turbo",
        "path": "mlx-community/whisper-large-v3-turbo",
        "default_batch": 36,
        "description": "速度快约2倍，精度略低，推荐日常使用"
    },
    "Large-v3 (精准)": {
        "name": "large-v3",
        "path": "mlx-community/whisper-large-v3-mlx",
        "default_batch": 12,
        "description": "精度最高，速度较慢，适合高要求场景"
    }
}

# 语言选项
LANGUAGES = {
    "自动检测": None,
    "中文": "zh",
    "英文": "en",
    "日文": "ja",
    "韩文": "ko",
    "法文": "fr",
    "德文": "de",
    "西班牙文": "es",
    "俄文": "ru",
}

# VAD 选项值映射（下拉菜单选项 -> 实际值）
VAD_OPTIONS = [
    ("禁用 VAD - 不跳过任何内容", None),
    ("极敏感 (0.2) - 轻声语音、耳语", 0.2),
    ("敏感 (0.3) - 安静环境、远场", 0.3),
    ("平衡 (0.5) - 默认，大多数场景", 0.5),
    ("严格 (0.7) - 有背景噪音", 0.7),
    ("极严格 (0.9) - 嘈杂环境", 0.9),
]
VAD_CHOICES = [opt[0] for opt in VAD_OPTIONS]
VAD_VALUE_MAP = {opt[0]: opt[1] for opt in VAD_OPTIONS}

# 预设场景 - 包含所有联动参数
PRESETS = {
    "📌 标准模式": {
        "description": "清晰录音、播客、采访、专业设备录制",
        "model": "Large-v3 (精准)",
        "batch_size": 12,
        "vad": "平衡 (0.5) - 默认，大多数场景",
        "enable_diarization": False,
        "num_speakers": None,
        "no_speech_threshold": 0.6,
        "logprob_threshold": -1.0,
        "compression_ratio_threshold": 2.4,
        "condition_on_previous_text": True,
        "temperature": 0.0,
    },
    "📞 电话/低音质": {
        "description": "电话录音、远程通话、低音量、手机录音 - 自动启用说话人分离",
        "model": "Large-v3 (精准)",
        "batch_size": 12,
        "vad": "禁用 VAD - 不跳过任何内容",
        "enable_diarization": True,
        "num_speakers": 2,
        "no_speech_threshold": 0.1,
        "logprob_threshold": -2.5,
        "compression_ratio_threshold": 5.0,
        "condition_on_previous_text": False,
        "temperature": 0.0,
    },
    "🏢 会议录音": {
        "description": "多人会议、讨论 - 自动启用说话人分离，需要上下文连贯",
        "model": "Large-v3 (精准)",
        "batch_size": 12,
        "vad": "敏感 (0.3) - 安静环境、远场",
        "enable_diarization": True,
        "num_speakers": None,
        "no_speech_threshold": 0.5,
        "logprob_threshold": -1.5,
        "compression_ratio_threshold": 3.0,
        "condition_on_previous_text": True,
        "temperature": 0.0,
    },
    "🎧 嘈杂环境": {
        "description": "户外录音、背景噪音大、咖啡厅等 - 严格过滤噪音",
        "model": "Large-v3 (精准)",
        "batch_size": 12,
        "vad": "严格 (0.7) - 有背景噪音",
        "enable_diarization": False,
        "num_speakers": None,
        "no_speech_threshold": 0.7,
        "logprob_threshold": -1.0,
        "compression_ratio_threshold": 2.4,
        "condition_on_previous_text": True,
        "temperature": 0.0,
    },
    "⚡ 快速转录": {
        "description": "使用 Turbo 模型，速度优先 - 适合长音频或批量处理",
        "model": "Large-v3-turbo (快速)",
        "batch_size": 36,
        "vad": "平衡 (0.5) - 默认，大多数场景",
        "enable_diarization": False,
        "num_speakers": None,
        "no_speech_threshold": 0.6,
        "logprob_threshold": -1.0,
        "compression_ratio_threshold": 2.4,
        "condition_on_previous_text": True,
        "temperature": 0.0,
    },
    "🔧 自定义": {
        "description": "手动调节所有参数 - 下方参数可自由调整",
        "model": "Large-v3 (精准)",
        "batch_size": 12,
        "vad": "平衡 (0.5) - 默认，大多数场景",
        "enable_diarization": False,
        "num_speakers": None,
        "no_speech_threshold": 0.6,
        "logprob_threshold": -1.0,
        "compression_ratio_threshold": 2.4,
        "condition_on_previous_text": True,
        "temperature": 0.0,
    }
}


# ========== 全局变量存储最新结果 ==========
LAST_RESULT = {
    "segments": [],
    "text": "",
    "audio_path": "",
    "audio_duration": 0,
    "model": "",
    "preset": "",
}


# ========== 核心转录函数 ==========

def format_time(seconds: float) -> str:
    """格式化时间为 HH:MM:SS.mmm"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"
    return f"{minutes:02d}:{secs:06.3f}"


def format_time_srt(seconds: float) -> str:
    """格式化时间为 SRT 格式 HH:MM:SS,mmm"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def transcribe_audio(
    audio_path: str,
    model_key: str,
    language_key: str,
    preset_key: str,
    enable_diarization: bool,
    num_speakers: Optional[int],
    hf_token: str,
    vad_key: str,
    batch_size: int,
    duration_minutes: Optional[float],
    # 高级参数
    no_speech_threshold: float,
    logprob_threshold: float,
    compression_ratio_threshold: float,
    condition_on_previous_text: bool,
    temperature: float,
    initial_prompt: str,
) -> Generator[Tuple[str, str, str], None, None]:
    """
    转录音频文件（生成器，实时输出日志）
    
    Yields:
        (log_text, result_text, status)
    """
    global LAST_RESULT
    
    logs = []
    
    def log(msg: str):
        timestamp = time.strftime('%H:%M:%S')
        logs.append(f"[{timestamp}] {msg}")
        return "\n".join(logs)
    
    try:
        # ===== 初始化 =====
        yield log("═" * 50), "", "⏳ 准备中..."
        yield log("🚀 MLX Whisper 语音转录启动"), "", "⏳ 准备中..."
        yield log("═" * 50), "", "⏳ 准备中..."
        
        # 获取配置
        model_config = MODELS[model_key]
        language = LANGUAGES[language_key]
        vad_threshold = VAD_VALUE_MAP.get(vad_key)
        
        yield log(""), "", "⏳ 准备中..."
        yield log(f"📁 音频文件: {Path(audio_path).name}"), "", "⏳ 准备中..."
        yield log(f"📂 完整路径: {audio_path}"), "", "⏳ 准备中..."
        yield log(""), "", "⏳ 准备中..."
        yield log(f"🤖 模型: {model_key}"), "", "⏳ 准备中..."
        yield log(f"   HuggingFace 仓库: {model_config['path']}"), "", "⏳ 准备中..."
        yield log(f"🌐 语言: {language_key} {'(自动检测)' if language is None else ''}"), "", "⏳ 准备中..."
        yield log(f"📋 预设模式: {preset_key}"), "", "⏳ 准备中..."
        
        yield log(""), "", "⏳ 准备中..."
        yield log("─" * 40), "", "⏳ 准备中..."
        yield log("📊 当前参数配置:"), "", "⏳ 准备中..."
        yield log("─" * 40), "", "⏳ 准备中..."
        yield log(f"   ⚡ Batch Size: {batch_size}"), "", "⏳ 准备中..."
        yield log(f"   🔊 VAD: {vad_key}"), "", "⏳ 准备中..."
        yield log(f"      → 实际值: {vad_threshold if vad_threshold else '禁用 (不跳过静音)'}"), "", "⏳ 准备中..."
        yield log(""), "", "⏳ 准备中..."
        yield log(f"   🎯 Whisper 高级参数:"), "", "⏳ 准备中..."
        yield log(f"      • no_speech_threshold = {no_speech_threshold}"), "", "⏳ 准备中..."
        yield log(f"        (无语音判断阈值: 越低保留越多语音)"), "", "⏳ 准备中..."
        yield log(f"      • logprob_threshold = {logprob_threshold}"), "", "⏳ 准备中..."
        yield log(f"        (置信度阈值: 越低接受越多低置信度结果)"), "", "⏳ 准备中..."
        yield log(f"      • compression_ratio_threshold = {compression_ratio_threshold}"), "", "⏳ 准备中..."
        yield log(f"        (压缩比阈值: 越高保留越多重复内容)"), "", "⏳ 准备中..."
        yield log(f"      • condition_on_previous_text = {condition_on_previous_text}"), "", "⏳ 准备中..."
        yield log(f"        (前文上下文: True=连贯但可能累积错误)"), "", "⏳ 准备中..."
        yield log(f"      • temperature = {temperature}"), "", "⏳ 准备中..."
        yield log(f"        (采样温度: 0.0=确定性输出)"), "", "⏳ 准备中..."
        
        if initial_prompt:
            yield log(f"      • initial_prompt = \"{initial_prompt[:40]}...\""), "", "⏳ 准备中..."
        
        yield log(""), "", "⏳ 准备中..."
        yield log(f"   👥 说话人分离: {'✅ 启用' if enable_diarization else '❌ 禁用'}"), "", "⏳ 准备中..."
        if enable_diarization:
            yield log(f"      → 说话人数量: {num_speakers if num_speakers else '自动检测'}"), "", "⏳ 准备中..."
        yield log("─" * 40), "", "⏳ 准备中..."
        
        # ===== 加载音频 =====
        yield log(""), "", "📂 加载音频..."
        yield log("📂 正在加载音频文件..."), "", "📂 加载音频..."
        
        load_start = time.time()
        
        # 确定加载时长
        if duration_minutes and duration_minutes > 0:
            duration_seconds = duration_minutes * 60
            yield log(f"   ⏱️ 只加载前 {duration_minutes} 分钟 ({duration_seconds:.0f} 秒)"), "", "📂 加载音频..."
            audio, sr = librosa.load(audio_path, sr=16000, mono=True, duration=duration_seconds)
        else:
            yield log(f"   📁 加载完整音频"), "", "📂 加载音频..."
            audio, sr = librosa.load(audio_path, sr=16000, mono=True)
        
        audio_duration = len(audio) / sr
        load_time = time.time() - load_start
        
        yield log(f"   ✅ 加载完成!"), "", "📂 加载音频..."
        yield log(f"   📊 音频时长: {audio_duration:.1f} 秒 ({audio_duration/60:.1f} 分钟)"), "", "📂 加载音频..."
        yield log(f"   ⏱️ 加载耗时: {load_time:.2f} 秒"), "", "📂 加载音频..."
        
        # ===== 转录 =====
        yield log(""), "", "🎤 转录中..."
        yield log("🎤 开始 ASR 转录..."), "", "🎤 转录中..."
        yield log("   ⏳ 首次运行需要下载模型，请耐心等待..."), "", "🎤 转录中..."
        
        import mlx_whisper
        
        transcribe_start = time.time()
        
        # 构建转录参数
        transcribe_params = {
            "path_or_hf_repo": model_config["path"],
            "language": language,
            "word_timestamps": True,
            "verbose": False,
            "no_speech_threshold": no_speech_threshold,
            "logprob_threshold": logprob_threshold,
            "compression_ratio_threshold": compression_ratio_threshold,
            "condition_on_previous_text": condition_on_previous_text,
            "temperature": temperature,
        }
        
        if initial_prompt:
            transcribe_params["initial_prompt"] = initial_prompt
        
        yield log("   🔄 模型加载中..."), "", "🎤 转录中..."
        
        result = mlx_whisper.transcribe(audio, **transcribe_params)
        
        transcribe_time = time.time() - transcribe_start
        realtime_ratio = audio_duration / transcribe_time if transcribe_time > 0 else 0
        
        segments = result.get('segments', [])
        text = result.get('text', '')
        
        yield log(f"   ✅ 转录完成!"), "", "🎤 转录完成"
        yield log(f"   ⏱️ 处理时间: {transcribe_time:.1f} 秒"), "", "🎤 转录完成"
        yield log(f"   📊 实时比: {realtime_ratio:.2f}x (>1 表示比实时快)"), "", "🎤 转录完成"
        yield log(f"   📝 识别段落: {len(segments)} 个"), "", "🎤 转录完成"
        yield log(f"   📄 文本长度: {len(text)} 字符"), "", "🎤 转录完成"
        
        # ===== 说话人分离 =====
        speaker_segments = []
        if enable_diarization:
            yield log(""), "", "👥 说话人分离中..."
            yield log("👥 开始说话人分离..."), "", "👥 说话人分离中..."
            
            diarize_start = time.time()
            
            try:
                from src.diarization import SpeakerDiarizer
                
                token = hf_token.strip() if hf_token else None
                
                if not token:
                    yield log("   ⚠️ 未提供 HuggingFace Token"), "", "👥 说话人分离中..."
                    yield log("   ⚠️ 将使用简化的能量分离方法"), "", "👥 说话人分离中..."
                
                yield log(f"   🔄 初始化说话人分离模型..."), "", "👥 说话人分离中..."
                yield log(f"   📊 目标说话人数: {num_speakers if num_speakers else '自动检测'}"), "", "👥 说话人分离中..."
                
                diarizer = SpeakerDiarizer(
                    num_speakers=num_speakers if num_speakers and num_speakers > 0 else None,
                    min_speakers=1,
                    max_speakers=10,
                    hf_token=token
                )
                
                speaker_segments = diarizer.diarize(audio, sr)
                diarize_time = time.time() - diarize_start
                
                unique_speakers = set(s[0] for s in speaker_segments)
                yield log(f"   ✅ 说话人分离完成!"), "", "👥 说话人分离完成"
                yield log(f"   ⏱️ 处理时间: {diarize_time:.1f} 秒"), "", "👥 说话人分离完成"
                yield log(f"   📊 检测到 {len(unique_speakers)} 个说话人: {', '.join(sorted(unique_speakers))}"), "", "👥 说话人分离完成"
                
            except Exception as e:
                yield log(f"   ⚠️ 说话人分离失败: {str(e)}"), "", "👥 说话人分离失败"
                yield log(f"   📝 将使用无说话人标签的结果"), "", "👥 说话人分离失败"
        
        # ===== 生成结果 =====
        yield log(""), "", "📝 生成结果..."
        yield log("📝 生成转录结果..."), "", "📝 生成结果..."
        
        # 处理段落，添加说话人标签
        processed_segments = []
        raw_speakers = []  # 原始说话人ID列表（保持顺序）
        
        for seg in segments:
            seg_start = seg.get('start', 0)
            seg_end = seg.get('end', 0)
            seg_text = seg.get('text', '').strip()
            
            if not seg_text:
                continue
            
            speaker = None
            if enable_diarization and speaker_segments:
                # 找重叠最多的说话人
                best_speaker = "UNKNOWN"
                best_overlap = 0
                
                for speaker_id, spk_start, spk_end in speaker_segments:
                    overlap_start = max(seg_start, spk_start)
                    overlap_end = min(seg_end, spk_end)
                    overlap = max(0, overlap_end - overlap_start)
                    
                    if overlap > best_overlap:
                        best_overlap = overlap
                        best_speaker = speaker_id
                
                speaker = best_speaker
                if speaker and speaker not in raw_speakers:
                    raw_speakers.append(speaker)
            
            processed_segments.append({
                "start": seg_start,
                "end": seg_end,
                "text": seg_text,
                "speaker": speaker
            })
        
        # 保存结果到全局变量
        LAST_RESULT["segments"] = processed_segments
        LAST_RESULT["text"] = text
        LAST_RESULT["audio_path"] = audio_path
        LAST_RESULT["audio_duration"] = audio_duration
        LAST_RESULT["model"] = model_key
        LAST_RESULT["preset"] = preset_key
        
        # 生成显示文本 - 完整格式（带时间戳）
        result_lines = []
        result_lines.append(f"# 转录结果")
        result_lines.append(f"音频: {Path(audio_path).name}")
        result_lines.append(f"时长: {audio_duration:.1f}s ({audio_duration/60:.1f}分钟)")
        result_lines.append(f"模型: {model_key}")
        result_lines.append(f"耗时: {transcribe_time:.1f}s (实时比: {realtime_ratio:.1f}x)")
        result_lines.append("")
        result_lines.append("─" * 40)
        
        for seg in processed_segments:
            time_str = f"[{format_time(seg['start'])} → {format_time(seg['end'])}]"
            spk = seg.get('speaker')
            if spk:
                result_lines.append(f"{time_str} {spk}")
            else:
                result_lines.append(time_str)
            result_lines.append(seg['text'])
            result_lines.append("")
        
        result_text = "\n".join(result_lines)
        
        # ===== 完成 =====
        yield log(""), result_text, "📝 生成结果..."
        yield log("═" * 50), result_text, "📝 生成结果..."
        yield log("✅ 全部处理完成!"), result_text, "📝 生成结果..."
        yield log(f"   📊 总耗时: {time.time() - load_start:.1f} 秒"), result_text, "📝 生成结果..."
        yield log("═" * 50), result_text, "📝 生成结果..."
        
        final_status = f"✅ 完成 | 时长 {audio_duration:.0f}s | 耗时 {transcribe_time:.0f}s | 实时比 {realtime_ratio:.1f}x"
        yield "\n".join(logs), result_text, final_status
        
    except Exception as e:
        error_msg = str(e)
        yield log(""), "", f"❌ 错误"
        yield log("═" * 50), "", f"❌ 错误"
        yield log(f"❌ 发生错误: {error_msg}"), "", f"❌ 错误: {error_msg}"
        yield log(""), "", f"❌ 错误: {error_msg}"
        yield log("详细错误信息:"), "", f"❌ 错误: {error_msg}"
        yield log(traceback.format_exc()), "", f"❌ 错误: {error_msg}"


def apply_preset(preset_key: str):
    """
    应用预设，返回所有联动参数的新值
    """
    preset = PRESETS.get(preset_key, PRESETS["📌 标准模式"])
    
    # num_speakers 转为字符串，None 表示空
    num_speakers_str = ""
    if preset["num_speakers"] is not None:
        num_speakers_str = str(preset["num_speakers"])
    
    return (
        preset["model"],
        preset["batch_size"],
        preset["vad"],
        preset["enable_diarization"],
        num_speakers_str,
        preset["no_speech_threshold"],
        preset["logprob_threshold"],
        preset["compression_ratio_threshold"],
        preset["condition_on_previous_text"],
        preset["temperature"],
        f"**{preset_key}**: {preset['description']}"
    )


def update_model_batch(model_key: str):
    """更新模型对应的默认 batch size"""
    config = MODELS.get(model_key, list(MODELS.values())[0])
    return config["default_batch"]


def download_txt():
    """生成 TXT 下载文件 - 简洁格式：A: 内容"""
    if not LAST_RESULT["segments"]:
        return None
    
    # 收集所有说话人并映射到字母 A, B, C...
    speakers = []
    for seg in LAST_RESULT["segments"]:
        spk = seg.get('speaker')
        if spk and spk not in speakers:
            speakers.append(spk)
    
    # 创建映射：SPEAKER_00 -> A, SPEAKER_01 -> B, ...
    speaker_map = {}
    for i, spk in enumerate(speakers):
        speaker_map[spk] = chr(ord('A') + i)  # A, B, C, D...
    
    lines = []
    for seg in LAST_RESULT["segments"]:
        text = seg['text'].strip()
        if not text:
            continue
        
        spk = seg.get('speaker')
        if spk and spk in speaker_map:
            speaker_label = speaker_map[spk]
            lines.append(f"{speaker_label}: {text}")
        else:
            # 没有说话人标签时直接输出文本
            lines.append(text)
    
    # 写入临时文件
    audio_name = Path(LAST_RESULT['audio_path']).stem
    tmp_path = tempfile.mktemp(suffix=".txt", prefix=f"{audio_name}_")
    with open(tmp_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))
    
    return tmp_path


def download_json():
    """生成 JSON 下载文件"""
    if not LAST_RESULT["segments"]:
        return None
    
    data = {
        "audio_file": Path(LAST_RESULT['audio_path']).name,
        "audio_duration": LAST_RESULT['audio_duration'],
        "model": LAST_RESULT['model'],
        "preset": LAST_RESULT['preset'],
        "segments": LAST_RESULT["segments"]
    }
    
    audio_name = Path(LAST_RESULT['audio_path']).stem
    tmp_path = tempfile.mktemp(suffix=".json", prefix=f"{audio_name}_")
    with open(tmp_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    return tmp_path


def download_srt():
    """生成 SRT 下载文件 - 说话人用 A, B, C 表示"""
    if not LAST_RESULT["segments"]:
        return None
    
    # 收集所有说话人并映射到字母
    speakers = []
    for seg in LAST_RESULT["segments"]:
        spk = seg.get('speaker')
        if spk and spk not in speakers:
            speakers.append(spk)
    
    speaker_map = {spk: chr(ord('A') + i) for i, spk in enumerate(speakers)}
    
    lines = []
    for i, seg in enumerate(LAST_RESULT["segments"], 1):
        lines.append(str(i))
        lines.append(f"{format_time_srt(seg['start'])} --> {format_time_srt(seg['end'])}")
        
        spk = seg.get('speaker')
        if spk and spk in speaker_map:
            lines.append(f"[{speaker_map[spk]}] {seg['text']}")
        else:
            lines.append(seg['text'])
        lines.append("")
    
    audio_name = Path(LAST_RESULT['audio_path']).stem
    tmp_path = tempfile.mktemp(suffix=".srt", prefix=f"{audio_name}_")
    with open(tmp_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))
    
    return tmp_path


# ========== UI 构建 ==========

CSS = """
/* 日志框样式 - 终端风格 */
.log-box textarea {
    font-family: 'SF Mono', 'Monaco', 'Consolas', monospace !important;
    font-size: 12px !important;
    line-height: 1.4 !important;
    background-color: #1a1a2e !important;
    color: #00ff88 !important;
}

/* 预设信息样式 */
.preset-info {
    padding: 10px;
    border-left: 4px solid #4CAF50;
    background: rgba(76, 175, 80, 0.1);
    margin: 10px 0;
    border-radius: 0 8px 8px 0;
}
"""

def create_ui():
    """创建 Gradio UI"""
    
    with gr.Blocks(title="MLX Whisper 语音转录", css=CSS) as app:
        
        gr.Markdown("""
        # 🎤 MLX Whisper 语音转录
        
        基于 Apple Silicon 优化的本地语音转录工具，支持说话人分离。
        
        > **使用说明**: 选择预设场景会自动配置所有参数。如需微调，选择「自定义」模式后调整下方参数。
        """)
        
        with gr.Row():
            # ===== 左栏：配置 =====
            with gr.Column(scale=1):
                
                # 音频上传
                audio_input = gr.File(
                    label="📁 上传音频文件",
                    file_types=[".wav", ".mp3", ".m4a", ".flac", ".ogg", ".webm"],
                    file_count="single"
                )
                
                # 转录时长（单文件时可用）- 默认空，不是0
                duration_minutes = gr.Textbox(
                    label="⏱️ 只转录前 N 分钟（留空=全部）",
                    value="",
                    placeholder="留空表示转录全部，输入数字如 2 表示前2分钟",
                    info="留空=转录全部内容"
                )
                
                gr.Markdown("---")
                
                # ===== 预设场景 (最高层级) =====
                gr.Markdown("### 🎯 第一步：选择场景预设")
                
                preset_dropdown = gr.Dropdown(
                    choices=list(PRESETS.keys()),
                    value="📌 标准模式",
                    label="场景预设",
                    info="选择场景后，下方所有参数会自动配置"
                )
                
                # 预设说明 (显著位置)
                preset_info = gr.Markdown(
                    value=f"**📌 标准模式**: {PRESETS['📌 标准模式']['description']}",
                    elem_classes=["preset-info"]
                )
                
                gr.Markdown("---")
                
                # ===== 基础参数 =====
                gr.Markdown("### ⚙️ 基础参数")
                
                with gr.Row():
                    model_dropdown = gr.Dropdown(
                        choices=list(MODELS.keys()),
                        value="Large-v3 (精准)",
                        label="🤖 模型",
                        info="Turbo 更快 (约2x)，Large-v3 更准"
                    )
                    
                    language_dropdown = gr.Dropdown(
                        choices=list(LANGUAGES.keys()),
                        value="自动检测",
                        label="🌐 语言",
                        info="指定语言可提高准确率"
                    )
                
                with gr.Row():
                    batch_size = gr.Slider(
                        minimum=4,
                        maximum=64,
                        step=4,
                        value=12,
                        label="⚡ Batch Size",
                        info="越大越快，8GB 内存建议 ≤24"
                    )
                    
                    vad_dropdown = gr.Dropdown(
                        choices=VAD_CHOICES,
                        value="平衡 (0.5) - 默认，大多数场景",
                        label="🔊 VAD (静音跳过)",
                        info="禁用 = 不跳过任何内容"
                    )
                
                gr.Markdown("---")
                
                # ===== 说话人分离 =====
                gr.Markdown("### 👥 说话人分离")
                
                enable_diarization = gr.Checkbox(
                    label="启用说话人分离",
                    value=False,
                    info="识别不同说话人，适合多人对话/会议/电话"
                )
                
                with gr.Row():
                    num_speakers = gr.Textbox(
                        label="说话人数量",
                        value="",
                        placeholder="留空=自动检测",
                        info="留空=自动检测"
                    )
                    
                    hf_token = gr.Textbox(
                        label="HuggingFace Token",
                        type="password",
                        placeholder="hf_...",
                        value=os.getenv("HF_TOKEN", ""),
                        info="首次使用需配置"
                    )
                
                gr.Markdown("---")
                
                # ===== 高级 Whisper 参数 (始终可见) =====
                gr.Markdown("### 🔧 高级 Whisper 参数")
                gr.Markdown("*这些参数会随预设自动变化，也可手动微调。选择「自定义」模式后可自由调整。*")
                
                # 第一行：核心阈值
                with gr.Row():
                    no_speech_threshold = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        step=0.05,
                        value=0.6,
                        label="no_speech_threshold",
                        info="无语音阈值 | 低(0.1)=电话 | 默认(0.6)=清晰 | 高(0.9)=过滤噪音"
                    )
                    
                    logprob_threshold = gr.Slider(
                        minimum=-3.0,
                        maximum=0.0,
                        step=0.1,
                        value=-1.0,
                        label="logprob_threshold",
                        info="置信度阈值 | 低(-2.5)=电话 | 默认(-1.0)=清晰"
                    )
                
                # 第二行：其他阈值
                with gr.Row():
                    compression_ratio_threshold = gr.Slider(
                        minimum=1.0,
                        maximum=10.0,
                        step=0.5,
                        value=2.4,
                        label="compression_ratio_threshold",
                        info="压缩比阈值 | 默认(2.4)=标准 | 高(5.0)=电话/宽松"
                    )
                    
                    temperature = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        step=0.1,
                        value=0.0,
                        label="temperature",
                        info="采样温度 | 0.0=确定性 | 越高越随机"
                    )
                
                # 第三行：开关选项
                with gr.Row():
                    condition_on_previous_text = gr.Checkbox(
                        label="condition_on_previous_text",
                        value=True,
                        info="基于前文 | 开=连贯 | 关=减少重复幻听"
                    )
                
                # 初始提示词
                initial_prompt = gr.Textbox(
                    label="initial_prompt (初始提示词)",
                    placeholder="例如：这是一段电话录音，讨论工作内容。涉及的专有名词：...",
                    value="",
                    info="引导模型识别特定术语、人名、主题"
                )
                
                # 开始按钮
                submit_btn = gr.Button("🚀 开始转录", variant="primary", size="lg")
            
            # ===== 右栏：结果 =====
            with gr.Column(scale=1):
                
                status_text = gr.Textbox(
                    label="状态",
                    value="⏸️ 等待上传音频...",
                    interactive=False,
                    lines=1
                )
                
                log_output = gr.Textbox(
                    label="📋 实时日志",
                    lines=15,
                    interactive=False,
                    elem_classes=["log-box"]
                )
                
                result_output = gr.Textbox(
                    label="📝 转录结果",
                    value="转录结果将显示在这里...",
                    lines=15,
                    interactive=False
                )
                
                with gr.Row():
                    download_txt_btn = gr.DownloadButton("📥 TXT", size="sm", variant="primary")
                    download_json_btn = gr.DownloadButton("📥 JSON", size="sm")
                    download_srt_btn = gr.DownloadButton("📥 SRT", size="sm")
        
        # ===== 事件绑定 =====
        
        # 预设变更 - 联动所有参数
        preset_dropdown.change(
            fn=apply_preset,
            inputs=[preset_dropdown],
            outputs=[
                model_dropdown,
                batch_size,
                vad_dropdown,
                enable_diarization,
                num_speakers,
                no_speech_threshold,
                logprob_threshold,
                compression_ratio_threshold,
                condition_on_previous_text,
                temperature,
                preset_info
            ]
        )
        
        # 模型变更 - 更新 batch size
        model_dropdown.change(
            fn=update_model_batch,
            inputs=[model_dropdown],
            outputs=[batch_size]
        )
        
        # 下载按钮
        download_txt_btn.click(fn=download_txt, outputs=[download_txt_btn])
        download_json_btn.click(fn=download_json, outputs=[download_json_btn])
        download_srt_btn.click(fn=download_srt, outputs=[download_srt_btn])
        
        # 提交转录
        def on_submit(
            audio_file,
            model_key,
            language_key,
            preset_key,
            enable_diarization,
            num_speakers_str,
            hf_token,
            vad_key,
            batch_size,
            duration_minutes_str,
            no_speech_threshold,
            logprob_threshold,
            compression_ratio_threshold,
            condition_on_previous_text,
            temperature,
            initial_prompt,
        ):
            if audio_file is None:
                yield "请先上传音频文件", "请先上传音频文件", "❌ 请先上传音频文件"
                return
            
            # 获取音频路径
            if hasattr(audio_file, 'name'):
                audio_path = audio_file.name
            else:
                audio_path = str(audio_file)
            
            # 处理 num_speakers - 空字符串表示自动检测
            num_speakers = None
            if num_speakers_str and num_speakers_str.strip():
                try:
                    num_speakers = int(num_speakers_str.strip())
                    if num_speakers <= 0:
                        num_speakers = None
                except ValueError:
                    num_speakers = None
            
            # 处理 duration_minutes - 空字符串表示全部
            duration_minutes = None
            if duration_minutes_str and duration_minutes_str.strip():
                try:
                    duration_minutes = float(duration_minutes_str.strip())
                    if duration_minutes <= 0:
                        duration_minutes = None
                except ValueError:
                    duration_minutes = None
            
            # 调用转录函数
            for log_text, result_text, status in transcribe_audio(
                audio_path=audio_path,
                model_key=model_key,
                language_key=language_key,
                preset_key=preset_key,
                enable_diarization=enable_diarization,
                num_speakers=num_speakers,
                hf_token=hf_token,
                vad_key=vad_key,
                batch_size=int(batch_size),
                duration_minutes=duration_minutes,
                no_speech_threshold=no_speech_threshold,
                logprob_threshold=logprob_threshold,
                compression_ratio_threshold=compression_ratio_threshold,
                condition_on_previous_text=condition_on_previous_text,
                temperature=temperature,
                initial_prompt=initial_prompt,
            ):
                yield log_text, result_text, status
        
        submit_btn.click(
            fn=on_submit,
            inputs=[
                audio_input,
                model_dropdown,
                language_dropdown,
                preset_dropdown,
                enable_diarization,
                num_speakers,
                hf_token,
                vad_dropdown,
                batch_size,
                duration_minutes,
                no_speech_threshold,
                logprob_threshold,
                compression_ratio_threshold,
                condition_on_previous_text,
                temperature,
                initial_prompt
            ],
            outputs=[log_output, result_output, status_text]
        )
    
    return app


# ========== 主入口 ==========

if __name__ == "__main__":
    app = create_ui()
    print("\n" + "="*50)
    print("🚀 正在启动 UI 服务器...")
    print("="*50 + "\n")
    
    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False
    )
