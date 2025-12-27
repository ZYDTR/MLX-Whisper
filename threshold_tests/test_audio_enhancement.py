#!/usr/bin/env python3
"""
音频增强测试 - 专注于前30秒无法识别的部分

结合最佳参数 (temperature=0.2) 和各种音频增强技术
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
OUTPUT_DIR = Path(__file__).parent / "enhanced_results"
OUTPUT_DIR.mkdir(exist_ok=True)

# 最佳Whisper参数
BEST_WHISPER_PARAMS = {
    "path_or_hf_repo": "mlx-community/whisper-large-v3-mlx",
    "language": "zh",
    "word_timestamps": True,
    "verbose": False,
    "no_speech_threshold": 0.1,
    "logprob_threshold": -2.5,
    "temperature": 0.2,  # 最佳温度
}


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


# ==================== 音频增强方法 ====================

def enhance_spectral_subtraction(audio, sr=16000):
    """
    频谱减法降噪
    估计噪声频谱并从信号中减去
    """
    print("\n🔊 频谱减法降噪")
    
    # 使用前0.5秒估计噪声
    noise_samples = int(0.5 * sr)
    noise = audio[:noise_samples]
    
    # STFT参数
    n_fft = 2048
    hop_length = 512
    
    # 计算噪声频谱
    noise_stft = librosa.stft(noise, n_fft=n_fft, hop_length=hop_length)
    noise_mag = np.abs(noise_stft).mean(axis=1, keepdims=True)
    
    # 计算信号STFT
    signal_stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    signal_mag = np.abs(signal_stft)
    signal_phase = np.angle(signal_stft)
    
    # 频谱减法
    alpha = 2.0  # 过减因子
    beta = 0.01  # 频谱下限
    enhanced_mag = np.maximum(signal_mag - alpha * noise_mag, beta * signal_mag)
    
    # 重构信号
    enhanced_stft = enhanced_mag * np.exp(1j * signal_phase)
    enhanced = librosa.istft(enhanced_stft, hop_length=hop_length, length=len(audio))
    
    return normalize(enhanced.astype(np.float32))


def enhance_wiener_filter(audio, sr=16000):
    """
    维纳滤波降噪
    """
    print("\n🔊 维纳滤波降噪")
    
    n_fft = 2048
    hop_length = 512
    
    # 估计噪声（前0.5秒）
    noise_samples = int(0.5 * sr)
    noise = audio[:noise_samples]
    noise_stft = librosa.stft(noise, n_fft=n_fft, hop_length=hop_length)
    noise_power = np.mean(np.abs(noise_stft) ** 2, axis=1, keepdims=True)
    
    # 信号STFT
    signal_stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    signal_power = np.abs(signal_stft) ** 2
    
    # 维纳滤波器
    snr = np.maximum(signal_power - noise_power, 0) / (noise_power + 1e-10)
    wiener_gain = snr / (snr + 1)
    
    # 应用滤波器
    enhanced_stft = signal_stft * wiener_gain
    enhanced = librosa.istft(enhanced_stft, hop_length=hop_length, length=len(audio))
    
    return normalize(enhanced.astype(np.float32))


def enhance_adaptive_gain(audio, sr=16000, target_rms=0.15, max_gain=30, window_sec=0.3):
    """
    自适应增益 - 动态调整音量
    """
    print(f"\n🔊 自适应增益 (target_rms={target_rms}, max_gain={max_gain}x)")
    
    window_samples = int(window_sec * sr)
    num_windows = len(audio) // window_samples + 1
    gains = np.ones(len(audio), dtype=np.float32)
    
    for i in range(num_windows):
        start = i * window_samples
        end = min(start + window_samples, len(audio))
        if start >= len(audio):
            break
        
        window = audio[start:end]
        rms = np.sqrt(np.mean(window ** 2)) + 1e-10
        
        # 噪声门控
        if rms < 0.001:
            gain = 1.0
        else:
            gain = min(target_rms / rms, max_gain)
        
        gains[start:end] = gain
    
    # 平滑
    gains = uniform_filter1d(gains, size=int(0.05 * sr))
    result = audio * gains
    
    # 软限幅
    result = np.tanh(result * 1.5) / 1.5 + result * 0.3
    
    return normalize(result.astype(np.float32))


def enhance_highpass_and_boost(audio, sr=16000, cutoff=150, boost_db=6):
    """
    高通滤波 + 人声频段增强
    """
    print(f"\n🔊 高通滤波({cutoff}Hz) + 人声增强(+{boost_db}dB)")
    
    nyq = sr / 2
    
    # 高通滤波去除低频噪声
    b, a = signal.butter(4, cutoff / nyq, btype='high')
    audio = signal.filtfilt(b, a, audio).astype(np.float32)
    
    # 人声频段增强 (300-3000Hz)
    b, a = signal.butter(4, [300 / nyq, 3000 / nyq], btype='band')
    voice_band = signal.filtfilt(b, a, audio)
    
    boost_linear = 10 ** (boost_db / 20)
    enhanced = audio + voice_band * (boost_linear - 1)
    
    return normalize(enhanced.astype(np.float32))


def enhance_compression_aggressive(audio, sr=16000):
    """
    激进的动态范围压缩
    """
    print("\n🔊 激进动态范围压缩")
    
    # 多级压缩
    result = audio.copy()
    
    configs = [
        (-35, 4.0, 10),  # threshold_db, ratio, makeup_db
        (-30, 3.0, 8),
        (-25, 2.5, 6),
    ]
    
    for threshold_db, ratio, makeup_db in configs:
        eps = 1e-10
        audio_db = 20 * np.log10(np.abs(result) + eps)
        
        gain_db = np.zeros_like(audio_db)
        above_threshold = audio_db > threshold_db
        gain_db[above_threshold] = (threshold_db - audio_db[above_threshold]) * (1 - 1/ratio)
        gain_db += makeup_db
        
        # 平滑
        gain_db = uniform_filter1d(gain_db, size=int(0.01 * sr))
        gain_linear = 10 ** (gain_db / 20)
        result = result * gain_linear
        
        # 软限幅
        result = np.tanh(result)
    
    return normalize(result.astype(np.float32))


def enhance_combined_best(audio, sr=16000):
    """
    组合最佳增强：高通 + 自适应增益 + 轻度压缩
    """
    print("\n🔊 组合增强 (高通 + 自适应增益 + 压缩)")
    
    # 1. 高通滤波
    nyq = sr / 2
    b, a = signal.butter(4, 100 / nyq, btype='high')
    result = signal.filtfilt(b, a, audio).astype(np.float32)
    
    # 2. 自适应增益
    result = enhance_adaptive_gain(result, sr, target_rms=0.12, max_gain=25, window_sec=0.25)
    
    # 3. 轻度压缩
    eps = 1e-10
    audio_db = 20 * np.log10(np.abs(result) + eps)
    gain_db = np.zeros_like(audio_db)
    threshold_db = -25
    ratio = 2.5
    above_threshold = audio_db > threshold_db
    gain_db[above_threshold] = (threshold_db - audio_db[above_threshold]) * (1 - 1/ratio)
    gain_db += 6
    gain_db = uniform_filter1d(gain_db, size=int(0.01 * sr))
    gain_linear = 10 ** (gain_db / 20)
    result = result * gain_linear
    
    return normalize(result.astype(np.float32))


def enhance_pre_emphasis(audio, sr=16000, coef=0.97):
    """
    预加重 - 提升高频，常用于语音处理
    """
    print(f"\n🔊 预加重 (coef={coef})")
    
    # 预加重滤波
    emphasized = np.append(audio[0], audio[1:] - coef * audio[:-1])
    
    # 自适应增益
    emphasized = enhance_adaptive_gain(emphasized.astype(np.float32), sr, 
                                        target_rms=0.12, max_gain=20)
    
    return normalize(emphasized.astype(np.float32))


# ==================== 转录和保存 ====================

def format_time(seconds):
    mins = int(seconds // 60)
    secs = seconds % 60
    return f"{mins:02d}:{secs:06.3f}"


def transcribe_and_save(audio, name, params=None):
    """转录并保存"""
    if params is None:
        params = BEST_WHISPER_PARAMS.copy()
    
    print(f"\n{'='*70}")
    print(f"🧪 转录: {name}")
    print("="*70)
    
    start_time = time.time()
    result = mlx_whisper.transcribe(audio, **params)
    elapsed = time.time() - start_time
    
    segments = result.get('segments', [])
    text = result.get('text', '')
    
    print(f"   ⏱️ 耗时: {elapsed:.1f}s")
    print(f"   📊 段落数: {len(segments)}")
    print(f"   📝 文本长度: {len(text)}")
    
    # 统计前30秒的识别情况
    early_segments = [s for s in segments if s.get('start', 999) < 30]
    print(f"   📍 前30秒段落数: {len(early_segments)}")
    
    # 保存
    output_file = OUTPUT_DIR / f"enhanced_{name}.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"方法: {name}\n")
        f.write(f"段落数: {len(segments)}, 前30秒段落: {len(early_segments)}\n")
        f.write("="*70 + "\n\n")
        
        for seg in segments:
            start = format_time(seg.get('start', 0))
            end = format_time(seg.get('end', 0))
            seg_text = seg.get('text', '').strip()
            if seg_text:
                f.write(f"[{start} -> {end}]\n{seg_text}\n\n")
    
    print(f"   💾 已保存: {output_file.name}")
    
    # 预览前30秒
    print(f"\n   📋 前30秒预览:")
    for seg in early_segments[:10]:
        start = format_time(seg.get('start', 0))
        seg_text = seg.get('text', '').strip()
        if seg_text:
            print(f"      [{start}] {seg_text[:50]}")
    
    if not early_segments:
        print(f"      ⚠️ 前30秒无识别结果")
    
    return {
        'name': name,
        'segments': len(segments),
        'early_segments': len(early_segments),
        'text_length': len(text),
        'time': elapsed
    }


def main():
    print("="*70)
    print("🎯 音频增强测试 - 专注提升前30秒的识别")
    print("="*70)
    
    # 加载原始音频
    audio_original, sr = load_audio()
    
    # 准备增强方法
    enhancements = [
        ("原始_temp0.2", normalize(audio_original)),
        ("频谱减法", enhance_spectral_subtraction(audio_original, sr)),
        ("维纳滤波", enhance_wiener_filter(audio_original, sr)),
        ("自适应增益", enhance_adaptive_gain(audio_original, sr)),
        ("高通_人声增强", enhance_highpass_and_boost(audio_original, sr)),
        ("激进压缩", enhance_compression_aggressive(audio_original, sr)),
        ("组合增强", enhance_combined_best(audio_original, sr)),
        ("预加重", enhance_pre_emphasis(audio_original, sr)),
    ]
    
    all_results = []
    
    for name, enhanced_audio in enhancements:
        result = transcribe_and_save(enhanced_audio, name)
        all_results.append(result)
    
    # 额外测试：频谱减法 + turbo模型
    print("\n" + "#"*70)
    print("# 额外测试: 频谱减法 + turbo模型")
    print("#"*70)
    
    turbo_params = BEST_WHISPER_PARAMS.copy()
    turbo_params["path_or_hf_repo"] = "mlx-community/whisper-large-v3-turbo"
    
    enhanced_audio = enhance_spectral_subtraction(audio_original, sr)
    result = transcribe_and_save(enhanced_audio, "频谱减法_turbo", turbo_params)
    all_results.append(result)
    
    # 组合增强 + turbo
    enhanced_audio = enhance_combined_best(audio_original, sr)
    result = transcribe_and_save(enhanced_audio, "组合增强_turbo", turbo_params)
    all_results.append(result)
    
    # 汇总
    print("\n" + "="*70)
    print("📊 所有增强方法汇总")
    print("="*70)
    print(f"\n{'方法':<20} {'总段落':<10} {'前30秒段落':<12} {'文本长度':<10}")
    print("-"*60)
    
    for r in all_results:
        print(f"{r['name']:<20} {r['segments']:<10} {r['early_segments']:<12} {r['text_length']:<10}")
    
    # 找出前30秒识别最多的
    best = max(all_results, key=lambda x: x['early_segments'])
    print(f"\n✨ 前30秒识别最多: {best['name']} ({best['early_segments']}段)")
    
    # 保存汇总
    summary_file = OUTPUT_DIR / "enhancement_summary.txt"
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("音频增强测试汇总\n")
        f.write("="*60 + "\n\n")
        f.write(f"{'方法':<20} {'总段落':<10} {'前30秒段落':<12} {'文本长度':<10}\n")
        f.write("-"*60 + "\n")
        for r in all_results:
            f.write(f"{r['name']:<20} {r['segments']:<10} {r['early_segments']:<12} {r['text_length']:<10}\n")
        f.write(f"\n最佳前30秒: {best['name']} ({best['early_segments']}段)\n")
    
    print(f"\n💾 汇总已保存: {summary_file}")
    print("\n" + "="*70)
    print("测试完成！查看 enhanced_results/ 文件夹")
    print("="*70)


if __name__ == "__main__":
    main()

