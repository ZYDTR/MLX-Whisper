#!/usr/bin/env python3
"""
性能测试脚本 - 测试不同 batch_size 和并行配置

测试方案：
1. baseline:   batch_size=12, parallel=1
2. batch_18:   batch_size=18, parallel=1
3. parallel_2: batch_size=12, parallel=2
4. aggressive: batch_size=18, parallel=2

处理前10分钟音频，监控内存和速度
如果内存压力过大导致速度下降，停止后续测试
"""

import os
import sys
import time
import json
import subprocess
import numpy as np
import librosa
import mlx_whisper
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, List, Tuple
from scipy import signal
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing

# 配置
AUDIO_PATH = "/Users/zhengyidi/MLX/20251205 234222-BF444D4E_part_000.m4a"
DURATION = 600  # 前10分钟
OUTPUT_DIR = Path(__file__).parent / "results"
OUTPUT_DIR.mkdir(exist_ok=True)

# 性能阈值
MIN_REALTIME_RATIO = 5.0  # 如果实时比低于5x，认为内存压力过大


@dataclass
class TestConfig:
    """测试配置"""
    name: str
    batch_size: int
    parallel_audio: int
    model: str = "mlx-community/whisper-large-v3-turbo"
    language: str = "zh"
    # Whisper 参数
    no_speech_threshold: float = 0.1
    logprob_threshold: float = -2.5
    temperature: float = 0.2
    condition_on_previous_text: bool = False
    # 音频增强
    enable_enhancement: bool = True
    enhancement_alpha: float = 2.0
    # VAD 参数 (当前测试不使用 VAD，直接全量转录)
    use_vad: bool = False
    vad_threshold: float = 0.3
    min_speech_duration: float = 0.25
    min_silence_duration: float = 0.1


@dataclass
class TestResult:
    """测试结果"""
    config_name: str
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
        # 获取最大常驻内存
        usage = resource.getrusage(resource.RUSAGE_SELF)
        return usage.ru_maxrss / (1024 ** 3)  # macOS 返回字节
    except:
        return 0.0


def get_system_memory_pressure() -> str:
    """获取系统内存压力"""
    try:
        result = subprocess.run(
            ["memory_pressure"], 
            capture_output=True, 
            text=True,
            timeout=5
        )
        return result.stdout.strip()
    except:
        return "unknown"


def enhance_spectral_subtraction(audio: np.ndarray, sr: int = 16000, alpha: float = 2.0) -> np.ndarray:
    """频谱减法降噪"""
    noise_samples = int(0.5 * sr)
    noise = audio[:noise_samples]
    
    n_fft = 2048
    hop_length = 512
    
    noise_stft = librosa.stft(noise, n_fft=n_fft, hop_length=hop_length)
    noise_mag = np.abs(noise_stft).mean(axis=1, keepdims=True)
    
    signal_stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    signal_mag = np.abs(signal_stft)
    signal_phase = np.angle(signal_stft)
    
    beta = 0.01
    enhanced_mag = np.maximum(signal_mag - alpha * noise_mag, beta * signal_mag)
    
    enhanced_stft = enhanced_mag * np.exp(1j * signal_phase)
    enhanced = librosa.istft(enhanced_stft, hop_length=hop_length, length=len(audio))
    
    max_val = np.abs(enhanced).max()
    if max_val > 0:
        enhanced = enhanced / max_val
    
    return enhanced.astype(np.float32)


def load_and_enhance_audio(audio_path: str, duration: float, config: TestConfig) -> Tuple[np.ndarray, int]:
    """加载并增强音频"""
    print(f"  📂 加载音频 (前 {duration}s)...")
    audio, sr = librosa.load(audio_path, sr=16000, mono=True, duration=duration)
    audio = audio.astype(np.float32)
    
    if config.enable_enhancement:
        print(f"  🔊 频谱减法增强 (alpha={config.enhancement_alpha})...")
        audio = enhance_spectral_subtraction(audio, sr, config.enhancement_alpha)
    
    return audio, sr


def transcribe_audio(audio: np.ndarray, config: TestConfig) -> dict:
    """转录音频"""
    result = mlx_whisper.transcribe(
        audio,
        path_or_hf_repo=config.model,
        language=config.language,
        word_timestamps=True,
        verbose=False,
        no_speech_threshold=config.no_speech_threshold,
        logprob_threshold=config.logprob_threshold,
        temperature=config.temperature,
        condition_on_previous_text=config.condition_on_previous_text,
    )
    return result


def format_time(seconds: float) -> str:
    """格式化时间"""
    mins = int(seconds // 60)
    secs = seconds % 60
    return f"{mins:02d}:{secs:06.3f}"


def save_transcription(result: dict, config: TestConfig, output_path: Path):
    """保存转录结果"""
    segments = result.get('segments', [])
    text = result.get('text', '')
    
    # 保存 TXT
    txt_path = output_path / f"{config.name}_transcription.txt"
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(f"配置: {config.name}\n")
        f.write(f"模型: {config.model}\n")
        f.write(f"段落数: {len(segments)}\n")
        f.write("="*70 + "\n\n")
        for seg in segments:
            start = format_time(seg.get('start', 0))
            end = format_time(seg.get('end', 0))
            seg_text = seg.get('text', '').strip()
            if seg_text:
                f.write(f"[{start} -> {end}]\n{seg_text}\n\n")
    
    # 保存 JSON
    json_path = output_path / f"{config.name}_transcription.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump({
            "config": asdict(config),
            "segments": segments,
            "text": text
        }, f, ensure_ascii=False, indent=2)
    
    # 保存 SRT
    srt_path = output_path / f"{config.name}_transcription.srt"
    with open(srt_path, 'w', encoding='utf-8') as f:
        for i, seg in enumerate(segments, 1):
            start = seg.get('start', 0)
            end = seg.get('end', 0)
            text = seg.get('text', '').strip()
            if text:
                # SRT 时间格式: HH:MM:SS,mmm
                start_srt = f"{int(start//3600):02d}:{int((start%3600)//60):02d}:{int(start%60):02d},{int((start%1)*1000):03d}"
                end_srt = f"{int(end//3600):02d}:{int((end%3600)//60):02d}:{int(end%60):02d},{int((end%1)*1000):03d}"
                f.write(f"{i}\n{start_srt} --> {end_srt}\n{text}\n\n")
    
    print(f"  💾 已保存: {txt_path.name}, {json_path.name}, {srt_path.name}")


def run_single_test(config: TestConfig, audio: np.ndarray, duration: float) -> TestResult:
    """运行单个测试"""
    print(f"\n{'='*70}")
    print(f"🧪 测试: {config.name}")
    print(f"   batch_size={config.batch_size}, parallel={config.parallel_audio}")
    print("="*70)
    
    # 记录初始内存
    initial_memory = get_memory_usage_gb()
    
    try:
        # 开始计时
        start_time = time.time()
        
        # 转录
        print(f"  🎤 开始转录...")
        result = transcribe_audio(audio, config)
        
        # 计算时间
        processing_time = time.time() - start_time
        realtime_ratio = duration / processing_time
        
        # 获取峰值内存
        peak_memory = get_memory_usage_gb()
        
        # 统计结果
        segments = result.get('segments', [])
        text = result.get('text', '')
        
        print(f"  ⏱️  处理时间: {processing_time:.1f}s")
        print(f"  📊 实时比: {realtime_ratio:.1f}x")
        print(f"  📝 段落数: {len(segments)}, 文本长度: {len(text)}")
        print(f"  💾 峰值内存: {peak_memory:.2f} GB")
        
        # 保存结果
        save_transcription(result, config, OUTPUT_DIR)
        
        return TestResult(
            config_name=config.name,
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
        return TestResult(
            config_name=config.name,
            duration_seconds=duration,
            processing_time=0,
            realtime_ratio=0,
            num_segments=0,
            text_length=0,
            peak_memory_gb=get_memory_usage_gb(),
            success=False,
            error_message=str(e)
        )


def run_parallel_test(config: TestConfig, audio: np.ndarray, duration: float) -> TestResult:
    """
    运行分块测试 (模拟多音频场景)
    
    注意: MLX/Metal 不支持多线程并行访问 GPU，因此采用顺序处理分块
    但这仍然测试了处理多个较短音频的场景
    """
    print(f"\n{'='*70}")
    print(f"🧪 分块测试: {config.name}")
    print(f"   batch_size={config.batch_size}, chunks={config.parallel_audio}")
    print(f"   ⚠️ 注意: MLX/Metal 不支持真正的 GPU 并行，改为顺序分块处理")
    print("="*70)
    
    # 将音频分成多份
    chunk_duration = duration / config.parallel_audio
    chunk_samples = int(chunk_duration * 16000)
    
    chunks = []
    for i in range(config.parallel_audio):
        start = i * chunk_samples
        end = min((i + 1) * chunk_samples, len(audio))
        chunks.append((audio[start:end], i * chunk_duration))
    
    try:
        start_time = time.time()
        
        print(f"  🎤 顺序转录 {config.parallel_audio} 个音频块...")
        
        # 顺序处理每个块 (避免 Metal 并发问题)
        all_segments = []
        for i, (chunk, chunk_offset) in enumerate(chunks):
            print(f"     处理块 {i+1}/{len(chunks)}...")
            result = transcribe_audio(chunk, config)
            
            # 调整时间戳
            for seg in result.get('segments', []):
                seg['start'] += chunk_offset
                seg['end'] += chunk_offset
            all_segments.extend(result.get('segments', []))
        
        processing_time = time.time() - start_time
        realtime_ratio = duration / processing_time
        peak_memory = get_memory_usage_gb()
        
        # 合并结果
        all_segments.sort(key=lambda x: x.get('start', 0))
        text = ' '.join(seg.get('text', '') for seg in all_segments)
        
        print(f"  ⏱️  处理时间: {processing_time:.1f}s")
        print(f"  📊 实时比: {realtime_ratio:.1f}x")
        print(f"  📝 段落数: {len(all_segments)}, 文本长度: {len(text)}")
        print(f"  💾 峰值内存: {peak_memory:.2f} GB")
        
        # 保存结果
        merged_result = {"segments": all_segments, "text": text}
        save_transcription(merged_result, config, OUTPUT_DIR)
        
        return TestResult(
            config_name=config.name,
            duration_seconds=duration,
            processing_time=processing_time,
            realtime_ratio=realtime_ratio,
            num_segments=len(all_segments),
            text_length=len(text),
            peak_memory_gb=peak_memory,
            success=True
        )
        
    except Exception as e:
        print(f"  ❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return TestResult(
            config_name=config.name,
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
    print("🚀 性能测试 - Batch Size 和并行配置")
    print("="*70)
    print(f"📂 音频文件: {AUDIO_PATH}")
    print(f"⏱️  测试时长: {DURATION}s ({DURATION//60} 分钟)")
    print(f"📁 输出目录: {OUTPUT_DIR}")
    print()
    
    # 检查系统内存
    print("📊 系统内存状态:")
    memory_pressure = get_system_memory_pressure()
    print(f"   {memory_pressure[:200]}..." if len(memory_pressure) > 200 else f"   {memory_pressure}")
    print()
    
    # 定义测试配置
    test_configs = [
        # 1. 基准测试
        TestConfig(
            name="baseline_b12_p1",
            batch_size=12,
            parallel_audio=1,
        ),
        
        # 2. 提升 batch_size
        TestConfig(
            name="batch18_b18_p1",
            batch_size=18,
            parallel_audio=1,
        ),
        
        # 3. 双音频并行
        TestConfig(
            name="parallel2_b12_p2",
            batch_size=12,
            parallel_audio=2,
        ),
        
        # 4. 激进模式
        TestConfig(
            name="aggressive_b18_p2",
            batch_size=18,
            parallel_audio=2,
        ),
    ]
    
    # 加载音频（只加载一次）
    print("📂 加载音频...")
    base_config = test_configs[0]
    audio, sr = load_and_enhance_audio(AUDIO_PATH, DURATION, base_config)
    audio_duration = len(audio) / sr
    print(f"  ✅ 加载完成: {audio_duration:.1f}s, {len(audio)} samples")
    print()
    
    # 运行测试
    all_results = []
    stop_testing = False
    
    for config in test_configs:
        if stop_testing:
            print(f"\n⚠️ 跳过测试 {config.name} (内存压力过大)")
            continue
        
        # 根据并行数选择测试方法
        if config.parallel_audio > 1:
            result = run_parallel_test(config, audio, audio_duration)
        else:
            result = run_single_test(config, audio, audio_duration)
        
        all_results.append(result)
        
        # 检查是否需要停止
        if result.success and result.realtime_ratio < MIN_REALTIME_RATIO:
            print(f"\n⚠️ 实时比 {result.realtime_ratio:.1f}x 低于阈值 {MIN_REALTIME_RATIO}x")
            print("   检测到内存压力，停止后续测试")
            stop_testing = True
    
    # 汇总结果
    print("\n" + "="*70)
    print("📊 测试结果汇总")
    print("="*70)
    
    print(f"\n{'配置':<25} {'处理时间':<12} {'实时比':<10} {'段落数':<10} {'内存(GB)':<10} {'状态':<8}")
    print("-"*80)
    
    for r in all_results:
        status = "✅ 成功" if r.success else "❌ 失败"
        print(f"{r.config_name:<25} {r.processing_time:<12.1f} {r.realtime_ratio:<10.1f}x {r.num_segments:<10} {r.peak_memory_gb:<10.2f} {status:<8}")
    
    # 找出最佳配置
    successful = [r for r in all_results if r.success]
    if successful:
        best = max(successful, key=lambda x: x.realtime_ratio)
        print(f"\n✨ 最佳配置: {best.config_name}")
        print(f"   实时比: {best.realtime_ratio:.1f}x, 内存: {best.peak_memory_gb:.2f} GB")
    
    # 保存汇总
    summary = {
        "audio_file": AUDIO_PATH,
        "audio_duration": audio_duration,
        "test_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "results": [r.to_dict() for r in all_results],
        "best_config": best.config_name if successful else None
    }
    
    summary_path = OUTPUT_DIR / "performance_summary.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 汇总已保存: {summary_path}")
    print("\n" + "="*70)
    print("测试完成！")
    print("="*70)
    
    return all_results


if __name__ == "__main__":
    results = main()

