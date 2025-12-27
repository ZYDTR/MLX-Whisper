#!/usr/bin/env python3
"""
隔离测试脚本：测试更低的 Whisper 阈值
目标：找到最佳的 no_speech_threshold 和 logprob_threshold 组合
"""

import time
import numpy as np
import librosa
import mlx_whisper
from pathlib import Path

AUDIO_PATH = "/Users/zhengyidi/MLX/20251205 234222-BF444D4E_part_000.m4a"
DURATION = 180  # 前3分钟
OUTPUT_DIR = Path(__file__).parent

def load_audio():
    """加载音频"""
    print(f"📂 加载音频: {AUDIO_PATH}")
    audio, sr = librosa.load(AUDIO_PATH, sr=16000, mono=True, duration=DURATION)
    # 归一化
    audio = audio / (np.abs(audio).max() + 1e-10)
    print(f"  ✅ 时长: {len(audio)/16000:.1f}s")
    return audio.astype(np.float32)

def format_time(seconds):
    """格式化时间"""
    mins = int(seconds // 60)
    secs = seconds % 60
    return f"{mins:02d}:{secs:06.3f}"

def transcribe_with_params(audio, no_speech_threshold, logprob_threshold, name):
    """使用指定参数转录"""
    print(f"\n{'='*70}")
    print(f"🧪 测试: {name}")
    print(f"   no_speech_threshold = {no_speech_threshold}")
    print(f"   logprob_threshold = {logprob_threshold}")
    print("="*70)
    
    start_time = time.time()
    
    result = mlx_whisper.transcribe(
        audio,
        path_or_hf_repo="mlx-community/whisper-large-v3-mlx",
        language="zh",
        word_timestamps=True,
        verbose=False,
        no_speech_threshold=no_speech_threshold,
        logprob_threshold=logprob_threshold,
        compression_ratio_threshold=6.0
    )
    
    elapsed = time.time() - start_time
    segments = result.get('segments', [])
    
    print(f"   ⏱️ 耗时: {elapsed:.1f}s")
    print(f"   📊 段落数: {len(segments)}")
    print(f"   📝 文本长度: {len(result.get('text', ''))}")
    
    # 保存结果
    output_file = OUTPUT_DIR / f"result_{name}.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"方法: {name}\n")
        f.write(f"参数: no_speech_threshold={no_speech_threshold}, logprob_threshold={logprob_threshold}\n")
        f.write(f"段落数: {len(segments)}, 文本长度: {len(result.get('text', ''))}\n")
        f.write("="*70 + "\n\n")
        
        for seg in segments:
            start = format_time(seg.get('start', 0))
            end = format_time(seg.get('end', 0))
            text = seg.get('text', '').strip()
            if text:  # 跳过空文本
                f.write(f"[{start} -> {end}]\n{text}\n\n")
    
    print(f"   💾 已保存: {output_file.name}")
    
    # 打印前10段预览
    print(f"\n   📋 预览 (前10段):")
    for seg in segments[:10]:
        start = format_time(seg.get('start', 0))
        text = seg.get('text', '').strip()
        if text:
            print(f"      [{start}] {text[:50]}{'...' if len(text) > 50 else ''}")
    
    return {
        'name': name,
        'no_speech_threshold': no_speech_threshold,
        'logprob_threshold': logprob_threshold,
        'segments': len(segments),
        'text_length': len(result.get('text', '')),
        'time': elapsed,
        'output_file': output_file.name
    }

def main():
    print("="*70)
    print("🎯 Whisper 阈值测试 - 寻找最佳低音量语音识别参数")
    print("="*70)
    
    # 加载音频
    audio = load_audio()
    
    # 测试配置：从当前最佳往更低调
    test_configs = [
        # 基准线（之前的最佳）
        (0.1, -2.5, "baseline_0.1_-2.5"),
        
        # 逐步降低 no_speech_threshold
        (0.05, -2.5, "nst_0.05"),
        (0.02, -2.5, "nst_0.02"),
        (0.01, -2.5, "nst_0.01"),
        
        # 逐步降低 logprob_threshold
        (0.1, -3.0, "lpt_-3.0"),
        (0.1, -3.5, "lpt_-3.5"),
        (0.1, -4.0, "lpt_-4.0"),
        
        # 组合更低的值
        (0.05, -3.0, "combo_0.05_-3.0"),
        (0.02, -3.5, "combo_0.02_-3.5"),
        (0.01, -4.0, "extreme_0.01_-4.0"),
    ]
    
    all_results = []
    
    for no_speech_th, logprob_th, name in test_configs:
        result = transcribe_with_params(audio, no_speech_th, logprob_th, name)
        all_results.append(result)
    
    # 汇总表格
    print("\n" + "="*70)
    print("📊 所有测试结果汇总")
    print("="*70)
    print(f"\n{'名称':<25} {'no_speech':<12} {'logprob':<10} {'段落数':<8} {'文本长度':<10}")
    print("-"*70)
    
    for r in all_results:
        print(f"{r['name']:<25} {r['no_speech_threshold']:<12} {r['logprob_threshold']:<10} {r['segments']:<8} {r['text_length']:<10}")
    
    # 保存汇总
    summary_file = OUTPUT_DIR / "summary.txt"
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("Whisper 阈值测试汇总\n")
        f.write("="*70 + "\n\n")
        f.write(f"{'名称':<25} {'no_speech':<12} {'logprob':<10} {'段落数':<8} {'文本长度':<10}\n")
        f.write("-"*70 + "\n")
        for r in all_results:
            f.write(f"{r['name']:<25} {r['no_speech_threshold']:<12} {r['logprob_threshold']:<10} {r['segments']:<8} {r['text_length']:<10}\n")
    
    print(f"\n💾 汇总已保存: {summary_file}")
    print("\n" + "="*70)
    print("测试完成！请对比各个 result_*.txt 文件")
    print("="*70)

if __name__ == "__main__":
    main()

