#!/usr/bin/env python3
"""
全面测试脚本：尝试所有未测试过的参数和模型

之前已测试的：
- no_speech_threshold: 0.1 -> 0.01
- logprob_threshold: -2.5 -> -4.0  
- compression_ratio_threshold: 6.0
- 各种音频增强方法

本次新测试：
1. 新模型：whisper-large-v3-turbo
2. condition_on_previous_text=False（可能减少重复/幻听）
3. initial_prompt（提供上下文提示）
4. 不同的 temperature 设置
5. hallucination_silence_threshold
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
    audio = audio / (np.abs(audio).max() + 1e-10)
    print(f"  ✅ 时长: {len(audio)/16000:.1f}s")
    return audio.astype(np.float32)

def format_time(seconds):
    mins = int(seconds // 60)
    secs = seconds % 60
    return f"{mins:02d}:{secs:06.3f}"

def transcribe_and_save(audio, name, **params):
    """转录并保存结果"""
    print(f"\n{'='*70}")
    print(f"🧪 测试: {name}")
    for k, v in params.items():
        if k != 'path_or_hf_repo':
            print(f"   {k} = {v}")
    print("="*70)
    
    start_time = time.time()
    
    try:
        result = mlx_whisper.transcribe(audio, **params)
    except Exception as e:
        print(f"   ❌ 错误: {e}")
        return None
    
    elapsed = time.time() - start_time
    segments = result.get('segments', [])
    text = result.get('text', '')
    
    print(f"   ⏱️ 耗时: {elapsed:.1f}s")
    print(f"   📊 段落数: {len(segments)}")
    print(f"   📝 文本长度: {len(text)}")
    
    # 保存结果
    output_file = OUTPUT_DIR / f"comprehensive_{name}.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"方法: {name}\n")
        f.write(f"参数: {params}\n")
        f.write(f"段落数: {len(segments)}, 文本长度: {len(text)}\n")
        f.write("="*70 + "\n\n")
        
        for seg in segments:
            start = format_time(seg.get('start', 0))
            end = format_time(seg.get('end', 0))
            seg_text = seg.get('text', '').strip()
            if seg_text:
                f.write(f"[{start} -> {end}]\n{seg_text}\n\n")
    
    print(f"   💾 已保存: {output_file.name}")
    
    # 预览
    print(f"\n   📋 预览:")
    for seg in segments[:8]:
        start = format_time(seg.get('start', 0))
        seg_text = seg.get('text', '').strip()
        if seg_text:
            print(f"      [{start}] {seg_text[:60]}{'...' if len(seg_text) > 60 else ''}")
    
    return {
        'name': name,
        'segments': len(segments),
        'text_length': len(text),
        'time': elapsed
    }

def main():
    print("="*70)
    print("🎯 全面测试 - 尝试所有未测试的参数和模型")
    print("="*70)
    
    audio = load_audio()
    all_results = []
    
    # ========== 测试1: large-v3-turbo 模型 ==========
    print("\n" + "#"*70)
    print("# 新模型测试: whisper-large-v3-turbo")
    print("#"*70)
    
    result = transcribe_and_save(
        audio,
        name="turbo_default",
        path_or_hf_repo="mlx-community/whisper-large-v3-turbo",
        language="zh",
        word_timestamps=True,
        verbose=False
    )
    if result: all_results.append(result)
    
    # turbo + 低阈值
    result = transcribe_and_save(
        audio,
        name="turbo_low_threshold",
        path_or_hf_repo="mlx-community/whisper-large-v3-turbo",
        language="zh",
        word_timestamps=True,
        verbose=False,
        no_speech_threshold=0.1,
        logprob_threshold=-2.5
    )
    if result: all_results.append(result)
    
    # ========== 测试2: condition_on_previous_text=False ==========
    print("\n" + "#"*70)
    print("# 测试: condition_on_previous_text=False")
    print("#"*70)
    
    result = transcribe_and_save(
        audio,
        name="v3_no_condition",
        path_or_hf_repo="mlx-community/whisper-large-v3-mlx",
        language="zh",
        word_timestamps=True,
        verbose=False,
        no_speech_threshold=0.1,
        logprob_threshold=-2.5,
        condition_on_previous_text=False  # 新参数
    )
    if result: all_results.append(result)
    
    # ========== 测试3: initial_prompt 提供上下文 ==========
    print("\n" + "#"*70)
    print("# 测试: initial_prompt 提供上下文提示")
    print("#"*70)
    
    result = transcribe_and_save(
        audio,
        name="v3_with_prompt",
        path_or_hf_repo="mlx-community/whisper-large-v3-mlx",
        language="zh",
        word_timestamps=True,
        verbose=False,
        no_speech_threshold=0.1,
        logprob_threshold=-2.5,
        initial_prompt="这是一个电话录音，有两个人在对话。一个人声音比较大，另一个人声音比较小。"
    )
    if result: all_results.append(result)
    
    # ========== 测试4: 不同的 temperature ==========
    print("\n" + "#"*70)
    print("# 测试: 不同的 temperature 设置")
    print("#"*70)
    
    # temperature=0 (贪婪解码，最确定性)
    result = transcribe_and_save(
        audio,
        name="v3_temp_0",
        path_or_hf_repo="mlx-community/whisper-large-v3-mlx",
        language="zh",
        word_timestamps=True,
        verbose=False,
        no_speech_threshold=0.1,
        logprob_threshold=-2.5,
        temperature=0.0
    )
    if result: all_results.append(result)
    
    # temperature=0.2 (略微随机)
    result = transcribe_and_save(
        audio,
        name="v3_temp_0.2",
        path_or_hf_repo="mlx-community/whisper-large-v3-mlx",
        language="zh",
        word_timestamps=True,
        verbose=False,
        no_speech_threshold=0.1,
        logprob_threshold=-2.5,
        temperature=0.2
    )
    if result: all_results.append(result)
    
    # ========== 测试5: hallucination_silence_threshold ==========
    print("\n" + "#"*70)
    print("# 测试: hallucination_silence_threshold")
    print("#"*70)
    
    result = transcribe_and_save(
        audio,
        name="v3_halluc_threshold",
        path_or_hf_repo="mlx-community/whisper-large-v3-mlx",
        language="zh",
        word_timestamps=True,
        verbose=False,
        no_speech_threshold=0.1,
        logprob_threshold=-2.5,
        hallucination_silence_threshold=0.5  # 跳过可能的幻听
    )
    if result: all_results.append(result)
    
    # ========== 测试6: 组合最佳参数 ==========
    print("\n" + "#"*70)
    print("# 测试: 组合多个优化参数")
    print("#"*70)
    
    result = transcribe_and_save(
        audio,
        name="v3_combined_best",
        path_or_hf_repo="mlx-community/whisper-large-v3-mlx",
        language="zh",
        word_timestamps=True,
        verbose=False,
        no_speech_threshold=0.1,
        logprob_threshold=-2.5,
        compression_ratio_threshold=5.0,
        condition_on_previous_text=False,
        initial_prompt="这是一个电话录音，两个人在聊天，讨论照片和衣服。",
        temperature=0.0
    )
    if result: all_results.append(result)
    
    # turbo + 组合参数
    result = transcribe_and_save(
        audio,
        name="turbo_combined_best",
        path_or_hf_repo="mlx-community/whisper-large-v3-turbo",
        language="zh",
        word_timestamps=True,
        verbose=False,
        no_speech_threshold=0.1,
        logprob_threshold=-2.5,
        compression_ratio_threshold=5.0,
        condition_on_previous_text=False,
        initial_prompt="这是一个电话录音，两个人在聊天，讨论照片和衣服。",
        temperature=0.0
    )
    if result: all_results.append(result)
    
    # ========== 汇总 ==========
    print("\n" + "="*70)
    print("📊 所有测试结果汇总")
    print("="*70)
    print(f"\n{'名称':<30} {'段落数':<10} {'文本长度':<10} {'耗时':<10}")
    print("-"*70)
    
    for r in all_results:
        print(f"{r['name']:<30} {r['segments']:<10} {r['text_length']:<10} {r['time']:.1f}s")
    
    # 保存汇总
    summary_file = OUTPUT_DIR / "comprehensive_summary.txt"
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("全面测试汇总\n")
        f.write("="*70 + "\n\n")
        f.write(f"{'名称':<30} {'段落数':<10} {'文本长度':<10} {'耗时':<10}\n")
        f.write("-"*70 + "\n")
        for r in all_results:
            f.write(f"{r['name']:<30} {r['segments']:<10} {r['text_length']:<10} {r['time']:.1f}s\n")
    
    print(f"\n💾 汇总已保存: {summary_file}")
    print("\n" + "="*70)
    print("测试完成！")
    print("="*70)

if __name__ == "__main__":
    main()

