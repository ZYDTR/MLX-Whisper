#!/usr/bin/env python3
"""
最佳组合测试 - 频谱减法 + turbo + 不同参数

基于发现：频谱减法 + turbo 能识别前30秒
进一步优化参数组合
"""

import time
import numpy as np
import librosa
import mlx_whisper
from pathlib import Path
from scipy import signal
from scipy.ndimage import uniform_filter1d

AUDIO_PATH = "/Users/zhengyidi/MLX/20251205 234222-BF444D4E_part_000.m4a"
DURATION = 180  # 前3分钟
OUTPUT_DIR = Path(__file__).parent / "best_combination_results"
OUTPUT_DIR.mkdir(exist_ok=True)


def load_audio():
    """加载音频"""
    print(f"📂 加载音频: {AUDIO_PATH}")
    audio, sr = librosa.load(AUDIO_PATH, sr=16000, mono=True, duration=DURATION)
    print(f"  ✅ 时长: {len(audio)/16000:.1f}s")
    return audio.astype(np.float32), sr


def normalize(audio):
    """归一化"""
    max_val = np.abs(audio).max()
    if max_val > 0:
        return audio / max_val
    return audio


def enhance_spectral_subtraction(audio, sr=16000, alpha=2.0, noise_duration=0.5):
    """
    频谱减法降噪
    """
    noise_samples = int(noise_duration * sr)
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
    
    return normalize(enhanced.astype(np.float32))


def enhance_spectral_plus_gain(audio, sr=16000, target_rms=0.12, max_gain=20):
    """
    频谱减法 + 自适应增益
    """
    # 先频谱减法
    enhanced = enhance_spectral_subtraction(audio, sr, alpha=2.0)
    
    # 再自适应增益
    window_sec = 0.3
    window_samples = int(window_sec * sr)
    num_windows = len(enhanced) // window_samples + 1
    gains = np.ones(len(enhanced), dtype=np.float32)
    
    for i in range(num_windows):
        start = i * window_samples
        end = min(start + window_samples, len(enhanced))
        if start >= len(enhanced):
            break
        
        window = enhanced[start:end]
        rms = np.sqrt(np.mean(window ** 2)) + 1e-10
        
        if rms < 0.001:
            gain = 1.0
        else:
            gain = min(target_rms / rms, max_gain)
        
        gains[start:end] = gain
    
    gains = uniform_filter1d(gains, size=int(0.05 * sr))
    result = enhanced * gains
    result = np.tanh(result * 1.5) / 1.5 + result * 0.3
    
    return normalize(result.astype(np.float32))


def format_time(seconds):
    mins = int(seconds // 60)
    secs = seconds % 60
    return f"{mins:02d}:{secs:06.3f}"


def transcribe_and_save(audio, name, params):
    """转录并保存"""
    print(f"\n{'='*70}")
    print(f"🧪 转录: {name}")
    print(f"   参数: no_speech={params.get('no_speech_threshold')}, "
          f"logprob={params.get('logprob_threshold')}, "
          f"temp={params.get('temperature')}")
    print("="*70)
    
    start_time = time.time()
    result = mlx_whisper.transcribe(audio, **params)
    elapsed = time.time() - start_time
    
    segments = result.get('segments', [])
    text = result.get('text', '')
    
    print(f"   ⏱️ 耗时: {elapsed:.1f}s")
    print(f"   📊 段落数: {len(segments)}")
    print(f"   📝 文本长度: {len(text)}")
    
    # 统计前30秒
    early_segments = [s for s in segments if s.get('start', 999) < 30]
    early_content = [s for s in early_segments if s.get('text', '').strip() not in ['嗯', '哦', '啊', '呃']]
    
    print(f"   📍 前30秒段落数: {len(early_segments)}")
    print(f"   📍 前30秒有效内容: {len(early_content)}")
    
    # 保存
    output_file = OUTPUT_DIR / f"best_{name}.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"方法: {name}\n")
        f.write(f"参数:\n")
        for k, v in params.items():
            if k != 'path_or_hf_repo':
                f.write(f"  {k}: {v}\n")
        f.write(f"\n段落数: {len(segments)}, 前30秒: {len(early_segments)}, 有效内容: {len(early_content)}\n")
        f.write("="*70 + "\n\n")
        
        for seg in segments:
            start = format_time(seg.get('start', 0))
            end = format_time(seg.get('end', 0))
            seg_text = seg.get('text', '').strip()
            if seg_text:
                f.write(f"[{start} -> {end}]\n{seg_text}\n\n")
    
    print(f"   💾 已保存: {output_file.name}")
    
    # 预览前30秒有效内容
    print(f"\n   📋 前30秒有效内容预览:")
    for seg in early_content[:15]:
        start = format_time(seg.get('start', 0))
        seg_text = seg.get('text', '').strip()
        print(f"      [{start}] {seg_text[:60]}")
    
    if not early_content:
        print(f"      ⚠️ 前30秒无有效内容")
    
    return {
        'name': name,
        'segments': len(segments),
        'early_segments': len(early_segments),
        'early_content': len(early_content),
        'text_length': len(text),
        'time': elapsed,
        'params': params
    }


def main():
    print("="*70)
    print("🎯 最佳组合测试 - 频谱减法 + turbo + 参数优化")
    print("="*70)
    
    audio_original, sr = load_audio()
    
    # 准备增强后的音频
    print("\n📡 准备增强音频...")
    audio_spectral = enhance_spectral_subtraction(audio_original, sr)
    audio_spectral_gain = enhance_spectral_plus_gain(audio_original, sr)
    
    # 测试配置
    test_configs = [
        # 基线：turbo + 默认参数
        ("turbo_default", audio_spectral, {
            "path_or_hf_repo": "mlx-community/whisper-large-v3-turbo",
            "language": "zh",
            "word_timestamps": True,
            "verbose": False,
        }),
        
        # turbo + temperature=0.2 (之前最佳)
        ("turbo_temp0.2", audio_spectral, {
            "path_or_hf_repo": "mlx-community/whisper-large-v3-turbo",
            "language": "zh",
            "word_timestamps": True,
            "verbose": False,
            "temperature": 0.2,
        }),
        
        # turbo + 低阈值
        ("turbo_low_threshold", audio_spectral, {
            "path_or_hf_repo": "mlx-community/whisper-large-v3-turbo",
            "language": "zh",
            "word_timestamps": True,
            "verbose": False,
            "no_speech_threshold": 0.1,
            "logprob_threshold": -2.5,
        }),
        
        # turbo + 低阈值 + temp0.2
        ("turbo_combined", audio_spectral, {
            "path_or_hf_repo": "mlx-community/whisper-large-v3-turbo",
            "language": "zh",
            "word_timestamps": True,
            "verbose": False,
            "no_speech_threshold": 0.1,
            "logprob_threshold": -2.5,
            "temperature": 0.2,
        }),
        
        # turbo + 极低阈值
        ("turbo_ultra_low", audio_spectral, {
            "path_or_hf_repo": "mlx-community/whisper-large-v3-turbo",
            "language": "zh",
            "word_timestamps": True,
            "verbose": False,
            "no_speech_threshold": 0.05,
            "logprob_threshold": -3.0,
            "temperature": 0.2,
        }),
        
        # 频谱减法+增益 + turbo
        ("turbo_spectral_gain", audio_spectral_gain, {
            "path_or_hf_repo": "mlx-community/whisper-large-v3-turbo",
            "language": "zh",
            "word_timestamps": True,
            "verbose": False,
            "no_speech_threshold": 0.1,
            "logprob_threshold": -2.5,
            "temperature": 0.2,
        }),
        
        # large-v3 + 频谱减法 (对比)
        ("v3_spectral_temp0.2", audio_spectral, {
            "path_or_hf_repo": "mlx-community/whisper-large-v3-mlx",
            "language": "zh",
            "word_timestamps": True,
            "verbose": False,
            "no_speech_threshold": 0.1,
            "logprob_threshold": -2.5,
            "temperature": 0.2,
        }),
        
        # turbo + 无condition
        ("turbo_no_condition", audio_spectral, {
            "path_or_hf_repo": "mlx-community/whisper-large-v3-turbo",
            "language": "zh",
            "word_timestamps": True,
            "verbose": False,
            "no_speech_threshold": 0.1,
            "logprob_threshold": -2.5,
            "temperature": 0.2,
            "condition_on_previous_text": False,
        }),
        
        # turbo + prompt
        ("turbo_with_prompt", audio_spectral, {
            "path_or_hf_repo": "mlx-community/whisper-large-v3-turbo",
            "language": "zh",
            "word_timestamps": True,
            "verbose": False,
            "no_speech_threshold": 0.1,
            "logprob_threshold": -2.5,
            "temperature": 0.2,
            "initial_prompt": "这是一段电话录音,有两个人在对话,请识别所有说话内容,包括远端电话那头的声音",
        }),
    ]
    
    all_results = []
    
    for name, audio, params in test_configs:
        result = transcribe_and_save(audio, name, params)
        all_results.append(result)
    
    # 汇总
    print("\n" + "="*70)
    print("📊 最佳组合测试汇总")
    print("="*70)
    print(f"\n{'配置':<25} {'前30秒段落':<12} {'有效内容':<10} {'总文本':<10}")
    print("-"*70)
    
    for r in all_results:
        print(f"{r['name']:<25} {r['early_segments']:<12} {r['early_content']:<10} {r['text_length']:<10}")
    
    # 找出最佳
    best_by_content = max(all_results, key=lambda x: x['early_content'])
    print(f"\n✨ 前30秒有效内容最多: {best_by_content['name']} ({best_by_content['early_content']}个)")
    
    # 保存汇总
    summary_file = OUTPUT_DIR / "best_combination_summary.txt"
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("最佳组合测试汇总\n")
        f.write("="*70 + "\n\n")
        f.write(f"{'配置':<25} {'前30秒段落':<12} {'有效内容':<10} {'总文本':<10}\n")
        f.write("-"*70 + "\n")
        for r in all_results:
            f.write(f"{r['name']:<25} {r['early_segments']:<12} {r['early_content']:<10} {r['text_length']:<10}\n")
        f.write(f"\n最佳有效内容: {best_by_content['name']} ({best_by_content['early_content']}个)\n")
    
    print(f"\n💾 汇总已保存: {summary_file}")
    print("\n" + "="*70)
    print("测试完成！")
    print("="*70)


if __name__ == "__main__":
    main()

