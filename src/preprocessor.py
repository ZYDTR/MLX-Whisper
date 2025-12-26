"""阶段 0: 音频预处理模块

负责将任意格式的音频统一转换为:
- 采样率: 16000 Hz
- 声道: 单声道 (Mono)
- 格式: float32 PCM [-1.0, 1.0]
"""

import numpy as np
import soundfile as sf
import librosa
from pathlib import Path
from typing import Tuple, Optional


# 支持的音频格式
SUPPORTED_FORMATS = {'.wav', '.mp3', '.m4a', '.flac', '.ogg', '.webm', '.opus'}

# 目标采样率 (Whisper 和 Senko 统一要求)
TARGET_SAMPLE_RATE = 16000


def load_audio(
    file_path: str,
    target_sr: int = TARGET_SAMPLE_RATE,
    mono: bool = True,
    normalize: bool = True
) -> Tuple[np.ndarray, int]:
    """
    加载并预处理音频文件
    
    Args:
        file_path: 音频文件路径
        target_sr: 目标采样率 (默认 16000 Hz)
        mono: 是否转换为单声道 (默认 True)
        normalize: 是否归一化到 [-1.0, 1.0] (默认 True)
    
    Returns:
        Tuple[np.ndarray, int]: (音频数据 float32, 采样率)
    
    Raises:
        ValueError: 不支持的音频格式
        FileNotFoundError: 文件不存在
    """
    path = Path(file_path)
    
    # 检查文件存在性
    if not path.exists():
        raise FileNotFoundError(f"音频文件不存在: {file_path}")
    
    # 检查格式支持
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_FORMATS:
        raise ValueError(
            f"不支持的音频格式: {suffix}. "
            f"支持的格式: {', '.join(SUPPORTED_FORMATS)}"
        )
    
    # 使用 librosa 加载音频 (自动处理各种格式)
    # librosa.load 会自动:
    # 1. 重采样到指定采样率
    # 2. 转换为单声道 (如果 mono=True)
    # 3. 转换为 float32
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        audio, sr = librosa.load(
            file_path,
            sr=target_sr,
            mono=mono,
            dtype=np.float32
        )
    
    # 归一化
    if normalize:
        audio = normalize_audio(audio)
    
    return audio, sr


def normalize_audio(audio: np.ndarray) -> np.ndarray:
    """
    归一化音频到 [-1.0, 1.0] 范围
    
    Args:
        audio: 输入音频数组
    
    Returns:
        归一化后的音频数组
    """
    max_val = np.abs(audio).max()
    if max_val > 0:
        audio = audio / max_val
    return audio.astype(np.float32)


def get_audio_info(file_path: str) -> dict:
    """
    获取音频文件信息 (不加载完整数据)
    
    Args:
        file_path: 音频文件路径
    
    Returns:
        包含音频信息的字典
    """
    path = Path(file_path)
    
    if not path.exists():
        raise FileNotFoundError(f"音频文件不存在: {file_path}")
    
    try:
        # 尝试使用 soundfile 获取元信息
        info = sf.info(file_path)
        return {
            "file": path.name,
            "format": info.format,
            "subtype": info.subtype,
            "channels": info.channels,
            "sample_rate": info.samplerate,
            "frames": info.frames,
            "duration": info.duration,
            "file_size_mb": path.stat().st_size / (1024 * 1024)
        }
    except Exception:
        # soundfile 失败时，使用 librosa 加载获取信息
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            audio, sr = librosa.load(file_path, sr=None, mono=True)
        
        return {
            "file": path.name,
            "format": path.suffix[1:].upper(),
            "subtype": "unknown",
            "channels": 1,
            "sample_rate": sr,
            "frames": len(audio),
            "duration": len(audio) / sr,
            "file_size_mb": path.stat().st_size / (1024 * 1024)
        }


def chunk_audio(
    audio: np.ndarray,
    sample_rate: int,
    chunk_duration: float = 600.0,  # 10 分钟
    overlap_duration: float = 30.0   # 30 秒重叠
) -> list:
    """
    将长音频分割成重叠的块
    
    用于处理超过 30 分钟的长音频，避免内存溢出
    
    Args:
        audio: 音频数据
        sample_rate: 采样率
        chunk_duration: 每块时长 (秒)
        overlap_duration: 重叠时长 (秒)
    
    Returns:
        List of tuples: [(chunk_audio, start_time, end_time), ...]
    """
    total_samples = len(audio)
    total_duration = total_samples / sample_rate
    
    chunk_samples = int(chunk_duration * sample_rate)
    overlap_samples = int(overlap_duration * sample_rate)
    step_samples = chunk_samples - overlap_samples
    
    chunks = []
    start_sample = 0
    
    while start_sample < total_samples:
        end_sample = min(start_sample + chunk_samples, total_samples)
        chunk_audio = audio[start_sample:end_sample]
        
        start_time = start_sample / sample_rate
        end_time = end_sample / sample_rate
        
        chunks.append((chunk_audio, start_time, end_time))
        
        # 如果已经到末尾，退出
        if end_sample >= total_samples:
            break
            
        start_sample += step_samples
    
    return chunks


def validate_audio(audio: np.ndarray, sample_rate: int) -> dict:
    """
    验证音频数据的有效性
    
    Args:
        audio: 音频数据
        sample_rate: 采样率
    
    Returns:
        验证结果字典
    """
    issues = []
    
    # 检查数据类型
    if audio.dtype != np.float32:
        issues.append(f"数据类型应为 float32，当前为 {audio.dtype}")
    
    # 检查值范围
    min_val, max_val = audio.min(), audio.max()
    if min_val < -1.0 or max_val > 1.0:
        issues.append(f"值范围应在 [-1.0, 1.0]，当前为 [{min_val:.3f}, {max_val:.3f}]")
    
    # 检查采样率
    if sample_rate != TARGET_SAMPLE_RATE:
        issues.append(f"采样率应为 {TARGET_SAMPLE_RATE}，当前为 {sample_rate}")
    
    # 检查维度 (应为 1D)
    if audio.ndim != 1:
        issues.append(f"应为单声道 (1D)，当前维度为 {audio.ndim}")
    
    # 检查是否全静音
    if np.abs(audio).max() < 1e-6:
        issues.append("音频似乎是全静音的")
    
    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "stats": {
            "dtype": str(audio.dtype),
            "shape": audio.shape,
            "sample_rate": sample_rate,
            "duration": len(audio) / sample_rate,
            "min": float(min_val),
            "max": float(max_val),
            "mean": float(audio.mean()),
            "std": float(audio.std())
        }
    }

