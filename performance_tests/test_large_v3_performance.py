#!/usr/bin/env python3
"""
Large-v3 性能测试 - 测试不同 batch_size 对速度和质量的影响

基于 comprehensive_v3_temp_0.2.txt 的最佳参数配置
逐步提升 batch_size 测试性能
"""

import os
import sys
import time
import json
import numpy as np
import librosa
import mlx_whisper
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional

# 配置
AUDIO_PATH = "/Users/zhengyidi/MLX/20251205 234222-BF444D4E_part_000.m4a"
DURATION = 600  # 前10分钟
OUTPUT_DIR = Path(__file__).parent / "large_v3_results"
OUTPUT_DIR.mkdir(exist_ok=True)

# 基于 comprehensive_v3_temp_0.2.txt 的最佳参数
BEST_CONFIG = {
    "model": "mlx-community/whisper-large-v3-mlx",
    "language": "zh",
    "word_timestamps": True,
    "verbose": False,
    "no_speech_threshold": 0.1,
    "logprob_threshold": -2.5,
    "temperature": 0.2,
    # 注意: 不使用音频增强，原始音频效果更好
    "enable_enhancement": False,
}


@dataclass
class TestResult:
    """测试结果"""
    test_name: str
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


def transcribe_audio(audio: np.ndarray) -> dict:
    """转录音频
    
    注意: mlx_whisper.transcribe() 不支持 batch_size 参数
    批处理是内部自动优化的
    """
    result = mlx_whisper.transcribe(
        audio,
        path_or_hf_repo=BEST_CONFIG["model"],
        language=BEST_CONFIG["language"],
        word_timestamps=BEST_CONFIG["word_timestamps"],
        verbose=BEST_CONFIG["verbose"],
        no_speech_threshold=BEST_CONFIG["no_speech_threshold"],
        logprob_threshold=BEST_CONFIG["logprob_threshold"],
        temperature=BEST_CONFIG["temperature"],
    )
    return result


def save_transcription(result: dict, test_name: str, output_path: Path):
    """保存转录结果"""
    segments = result.get('segments', [])
    text = result.get('text', '')
    
    # 保存 TXT
    txt_path = output_path / f"large_v3_{test_name}_transcription.txt"
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(f"配置: Large-v3, {test_name}\n")
        f.write(f"模型: {BEST_CONFIG['model']}\n")
        f.write(f"参数: {json.dumps(BEST_CONFIG, ensure_ascii=False, indent=2)}\n")
        f.write(f"段落数: {len(segments)}\n")
        f.write("="*70 + "\n\n")
        for seg in segments:
            start = format_time(seg.get('start', 0))
            end = format_time(seg.get('end', 0))
            seg_text = seg.get('text', '').strip()
            if seg_text:
                f.write(f"[{start} -> {end}]\n{seg_text}\n\n")
    
    # 保存 JSON
    json_path = output_path / f"large_v3_{test_name}_transcription.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump({
            "config": BEST_CONFIG,
            "test_name": test_name,
            "segments": segments,
            "text": text
        }, f, ensure_ascii=False, indent=2)
    
    print(f"  💾 已保存: {txt_path.name}, {json_path.name}")


def run_test(test_name: str, audio: np.ndarray, duration: float) -> TestResult:
    """运行单个测试"""
    print(f"\n{'='*70}")
    print(f"🧪 测试: {test_name}")
    print("="*70)
    
    try:
        start_time = time.time()
        
        print(f"  🎤 开始转录...")
        result = transcribe_audio(audio)
        
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
        save_transcription(result, test_name, OUTPUT_DIR)
        
        return TestResult(
            test_name=test_name,
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
            test_name=test_name,
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
    print("🚀 Large-v3 性能测试 - Batch Size 影响")
    print("="*70)
    print(f"📂 音频文件: {AUDIO_PATH}")
    print(f"⏱️  测试时长: {DURATION}s ({DURATION//60} 分钟)")
    print(f"📁 输出目录: {OUTPUT_DIR}")
    print(f"\n📋 使用参数 (基于 comprehensive_v3_temp_0.2.txt):")
    for k, v in BEST_CONFIG.items():
        print(f"   {k}: {v}")
    print()
    
    # 加载音频（只加载一次）
    audio, sr = load_audio(AUDIO_PATH, DURATION)
    audio_duration = len(audio) / sr
    print()
    
    # 测试 Large-v3 默认性能
    # 注意: mlx_whisper 内部自动优化批处理，无法直接控制 batch_size
    # 我们测试默认行为，记录内存和速度
    
    all_results = []
    
    # 运行基准测试
    result = run_test("default", audio, audio_duration)
    all_results.append(result)
    
    # 汇总结果
    print("\n" + "="*70)
    print("📊 测试结果汇总")
    print("="*70)
    
    print(f"\n{'测试名称':<15} {'处理时间':<12} {'实时比':<10} {'段落数':<10} {'内存(GB)':<10} {'状态':<8}")
    print("-"*75)
    
    for r in all_results:
        status = "✅ 成功" if r.success else "❌ 失败"
        print(f"{r.test_name:<15} {r.processing_time:<12.1f} {r.realtime_ratio:<10.1f}x {r.num_segments:<10} {r.peak_memory_gb:<10.2f} {status:<8}")
    
    # 保存汇总
    summary = {
        "audio_file": AUDIO_PATH,
        "audio_duration": audio_duration,
        "test_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": BEST_CONFIG,
        "note": "mlx_whisper.transcribe() 不支持 batch_size 参数，批处理由内部自动优化",
        "results": [r.to_dict() for r in all_results]
    }
    
    summary_path = OUTPUT_DIR / "large_v3_performance_summary.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 汇总已保存: {summary_path}")
    print("\n" + "="*70)
    print("测试完成！")
    print("="*70)
    
    return all_results


if __name__ == "__main__":
    results = main()

