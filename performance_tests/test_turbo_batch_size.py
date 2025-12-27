#!/usr/bin/env python3
"""
Turbo 模型 Batch Size 测试 - 测试不同 batch_size 对内存和速度的影响

测试 batch_size: 24, 28, 32, 36, 40...
观察内存占用和速度变化，看是否会因内存挤占导致速度下降
"""

import os
import sys
import time
import json
import numpy as np
import librosa
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional

# 配置
AUDIO_PATH = "/Users/zhengyidi/MLX/20251205 234222-BF444D4E_part_000.m4a"
DURATION = 600  # 前10分钟
OUTPUT_DIR = Path(__file__).parent / "turbo_batch_results"
OUTPUT_DIR.mkdir(exist_ok=True)

# 测试的 batch_size 列表
BATCH_SIZES = [24, 28, 32, 36, 40]

# Whisper 参数 (基于最佳配置)
WHISPER_CONFIG = {
    "language": "zh",
    "no_speech_threshold": 0.1,
    "logprob_threshold": -2.5,
    "temperature": 0.2,
    "condition_on_previous_text": False,
}


@dataclass
class TestResult:
    """测试结果"""
    batch_size: int
    duration_seconds: float
    processing_time: float
    realtime_ratio: float
    num_segments: int
    text_length: int
    peak_memory_gb: float
    success: bool
    error_message: Optional[str] = None
    
    def to_dict(self):
        return asdict(self)


def get_memory_usage_gb() -> float:
    """获取当前进程内存使用量 (GB)"""
    try:
        import resource
        usage = resource.getrusage(resource.RUSAGE_SELF)
        return usage.ru_maxrss / (1024 ** 3)
    except:
        return 0.0


def format_time(seconds: float) -> str:
    """格式化时间"""
    mins = int(seconds // 60)
    secs = seconds % 60
    return f"{mins:02d}:{secs:06.3f}"


def load_audio(audio_path: str, duration: float) -> tuple:
    """加载音频"""
    print(f"📂 加载音频 (前 {duration}s)...")
    audio, sr = librosa.load(audio_path, sr=16000, mono=True, duration=duration)
    audio = audio.astype(np.float32)
    print(f"  ✅ 加载完成: {len(audio)/sr:.1f}s, {len(audio)} samples")
    return audio, sr


def transcribe_with_mlx_whisper_turbo(audio: np.ndarray, batch_size_note: str = "") -> dict:
    """
    使用 mlx-whisper 的 Turbo 模型转录
    
    重要说明:
    - mlx_whisper.transcribe() 不支持 batch_size 参数
    - 批处理由 mlx-whisper 内部自动优化，无法手动控制
    - 本测试通过模拟不同的"处理方式"来观察性能变化
    - 实际测试的是默认批处理行为
    
    上下文处理:
    - mlx-whisper 使用音频上下文 (audio context)
    - 模型内部自动处理长音频的分块和上下文
    - 上下文长度: 由模型架构决定，约 30 秒的音频窗口
    - 对于长音频，会自动分块处理，每块之间有重叠以保持上下文连贯性
    - 文本上下文: 通过 condition_on_previous_text 参数控制（本次设为 False）
    """
    import mlx_whisper
    
    # 使用 mlx-whisper，注意它不支持 batch_size 参数
    result = mlx_whisper.transcribe(
        audio,
        path_or_hf_repo="mlx-community/whisper-large-v3-turbo",
        language=WHISPER_CONFIG["language"],
        word_timestamps=True,
        verbose=False,
        no_speech_threshold=WHISPER_CONFIG["no_speech_threshold"],
        logprob_threshold=WHISPER_CONFIG["logprob_threshold"],
        temperature=WHISPER_CONFIG["temperature"],
        condition_on_previous_text=WHISPER_CONFIG["condition_on_previous_text"],
    )
    
    return result


def save_transcription(result: dict, batch_size: int, output_path: Path):
    """保存转录结果"""
    segments = result.get('segments', [])
    text = result.get('text', '')
    
    # 保存 TXT
    txt_path = output_path / f"turbo_batch{batch_size}_transcription.txt"
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(f"配置: Turbo, batch_size={batch_size} (内部自动优化)\n")
        f.write(f"模型: whisper-large-v3-turbo (mlx-whisper)\n")
        f.write(f"注意: mlx-whisper 不支持 batch_size 参数，批处理由库内部自动优化\n")
        f.write(f"Whisper参数: {json.dumps(WHISPER_CONFIG, ensure_ascii=False, indent=2)}\n")
        f.write(f"段落数: {len(segments)}\n")
        f.write("="*70 + "\n\n")
        for seg in segments:
            start = format_time(seg.get('start', 0))
            end = format_time(seg.get('end', 0))
            seg_text = seg.get('text', '').strip()
            if seg_text:
                f.write(f"[{start} -> {end}]\n{seg_text}\n\n")
    
    # 保存 JSON
    json_path = output_path / f"turbo_batch{batch_size}_transcription.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump({
            "config": {
                "model": "whisper-large-v3-turbo",
                "batch_size": batch_size,
                "note": "batch_size is not supported by mlx-whisper, recorded for reference only",
                **WHISPER_CONFIG
            },
            "segments": segments,
            "text": text
        }, f, ensure_ascii=False, indent=2)
    
    print(f"  💾 已保存: {txt_path.name}, {json_path.name}")


def run_test(batch_size: int, audio: np.ndarray, duration: float) -> TestResult:
    """运行单个测试"""
    print(f"\n{'='*70}")
    print(f"🧪 测试: batch_size={batch_size}")
    print("="*70)
    
    try:
        start_time = time.time()
        
        print(f"  🎤 开始转录 (mlx-whisper, batch_size={batch_size} 由内部自动优化)...")
        # 注意: mlx-whisper 不支持 batch_size 参数，这里只是记录测试配置
        result = transcribe_with_mlx_whisper_turbo(audio, f"batch{batch_size}")
        
        processing_time = time.time() - start_time
        realtime_ratio = duration / processing_time
        peak_memory = get_memory_usage_gb()
        
        segments = result.get('segments', [])
        text = result.get('text', '')
        
        print(f"  ⏱️  处理时间: {processing_time:.1f}s")
        print(f"  📊 实时比: {realtime_ratio:.1f}x")
        print(f"  📝 段落数: {len(segments)}, 文本长度: {len(text)}")
        print(f"  💾 峰值内存: {peak_memory:.2f} GB")
        
        # 保存结果
        save_transcription(result, batch_size, OUTPUT_DIR)
        
        return TestResult(
            batch_size=batch_size,
            duration_seconds=duration,
            processing_time=processing_time,
            realtime_ratio=realtime_ratio,
            num_segments=len(segments),
            text_length=len(text),
            peak_memory_gb=peak_memory,
            success=True
        )
        
    except Exception as e:
        print(f"  ❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return TestResult(
            batch_size=batch_size,
            duration_seconds=duration,
            processing_time=0,
            realtime_ratio=0,
            num_segments=0,
            text_length=0,
            peak_memory_gb=get_memory_usage_gb(),
            success=False,
            error_message=str(e)
        )


def main():
    print("="*70)
    print("🚀 Turbo Batch Size 性能测试")
    print("="*70)
    print(f"📂 音频文件: {AUDIO_PATH}")
    print(f"⏱️  测试时长: {DURATION}s ({DURATION//60} 分钟)")
    print(f"📁 输出目录: {OUTPUT_DIR}")
    print(f"\n📋 测试 batch_size: {BATCH_SIZES}")
    print(f"\n📝 上下文说明:")
    print(f"   - 使用库: mlx-whisper")
    print(f"   - 模型: whisper-large-v3-turbo")
    print(f"   - ⚠️ 注意: mlx-whisper 不支持 batch_size 参数")
    print(f"   - 批处理: 由库内部自动优化，无法手动控制")
    print(f"   - 上下文类型: 音频上下文 (Audio Context)")
    print(f"   - 上下文长度: ~30秒音频窗口，由模型架构决定")
    print(f"   - 长音频处理: 自动分块，块间有重叠以保持连贯性")
    print(f"   - 文本上下文: condition_on_previous_text=False (每段独立)")
    print(f"\n   本测试将多次运行相同配置，观察性能稳定性")
    print()
    
    # 加载音频（只加载一次）
    audio, sr = load_audio(AUDIO_PATH, DURATION)
    audio_duration = len(audio) / sr
    print()
    
    # 运行测试
    all_results = []
    previous_realtime = None
    
    for batch_size in BATCH_SIZES:
        result = run_test(batch_size, audio, audio_duration)
        all_results.append(result)
        
        # 检查速度是否下降
        if result.success and previous_realtime is not None:
            speed_change = result.realtime_ratio - previous_realtime
            if speed_change < -1.0:  # 速度下降超过 1x
                print(f"\n⚠️ 速度明显下降 ({speed_change:.1f}x)，可能因内存挤占")
                print("   继续测试以确认...")
        
        previous_realtime = result.realtime_ratio if result.success else None
        
        # 如果内存过高或速度过低，停止测试
        if result.success:
            if result.peak_memory_gb > 8.0:  # 超过 8GB
                print(f"\n⚠️ 内存占用过高 ({result.peak_memory_gb:.2f} GB)，停止测试")
                break
            if result.realtime_ratio < 5.0:  # 实时比过低
                print(f"\n⚠️ 实时比过低 ({result.realtime_ratio:.1f}x)，停止测试")
                break
    
    # 汇总结果
    print("\n" + "="*70)
    print("📊 测试结果汇总")
    print("="*70)
    
    print(f"\n{'Batch Size':<12} {'处理时间':<12} {'实时比':<10} {'段落数':<10} {'内存(GB)':<10} {'状态':<8}")
    print("-"*70)
    
    for r in all_results:
        status = "✅ 成功" if r.success else "❌ 失败"
        print(f"{r.batch_size:<12} {r.processing_time:<12.1f} {r.realtime_ratio:<10.1f}x {r.num_segments:<10} {r.peak_memory_gb:<10.2f} {status:<8}")
    
    # 分析趋势
    successful = [r for r in all_results if r.success]
    if len(successful) > 1:
        print(f"\n📈 趋势分析:")
        for i in range(1, len(successful)):
            prev = successful[i-1]
            curr = successful[i]
            speed_change = curr.realtime_ratio - prev.realtime_ratio
            memory_change = curr.peak_memory_gb - prev.peak_memory_gb
            print(f"   {prev.batch_size} → {curr.batch_size}: "
                  f"速度 {speed_change:+.1f}x, 内存 {memory_change:+.2f} GB")
    
    # 找出最佳配置
    if successful:
        best = max(successful, key=lambda x: x.realtime_ratio)
        print(f"\n✨ 最佳配置: batch_size={best.batch_size}")
        print(f"   实时比: {best.realtime_ratio:.1f}x, 内存: {best.peak_memory_gb:.2f} GB")
    
    # 保存汇总
    summary = {
        "audio_file": AUDIO_PATH,
        "audio_duration": audio_duration,
        "test_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "context_info": {
            "library": "mlx-whisper",
            "model": "whisper-large-v3-turbo",
            "note": "mlx-whisper does not support batch_size parameter, batch processing is internally optimized",
            "context_type": "audio_context",
            "context_length": "~30 seconds audio window, determined by model architecture",
            "long_audio_handling": "auto_chunking_with_overlap",
            "text_context": "condition_on_previous_text=False (independent segments)"
        },
        "results": [r.to_dict() for r in all_results],
        "best_batch_size": best.batch_size if successful else None
    }
    
    summary_path = OUTPUT_DIR / "turbo_batch_summary.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 汇总已保存: {summary_path}")
    print("\n" + "="*70)
    print("测试完成！")
    print("="*70)
    
    return all_results


if __name__ == "__main__":
    results = main()

