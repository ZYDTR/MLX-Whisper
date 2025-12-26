"""阶段 1: VAD 语音活动检测模块

使用 Silero VAD (ONNX) 进行语音活动检测
- 极度轻量，毫秒级延迟
- 用于 ASR 分支，快速跳过静音段
"""

import numpy as np
from typing import List, Tuple, Optional
from .models import SpeechSegment
from .preprocessor import TARGET_SAMPLE_RATE


class VADDetector:
    """Silero VAD 语音活动检测器"""
    
    def __init__(
        self,
        threshold: float = 0.5,
        min_speech_duration: float = 0.25,
        min_silence_duration: float = 0.1,
        padding: float = 0.1,
        sample_rate: int = TARGET_SAMPLE_RATE
    ):
        """
        初始化 VAD 检测器
        
        Args:
            threshold: 语音概率阈值 (0-1)
            min_speech_duration: 最小语音片段时长 (秒)
            min_silence_duration: 最小静音时长，用于分割 (秒)
            padding: 片段前后填充时长 (秒)
            sample_rate: 采样率
        """
        self.threshold = threshold
        self.min_speech_duration = min_speech_duration
        self.min_silence_duration = min_silence_duration
        self.padding = padding
        self.sample_rate = sample_rate
        
        self._model = None
        self._utils = None
    
    def _load_model(self):
        """懒加载 Silero VAD 模型"""
        if self._model is None:
            import torch
            
            # 加载 Silero VAD 模型
            model, utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                onnx=True  # 使用 ONNX 版本，更快
            )
            
            self._model = model
            self._utils = utils
    
    def detect(self, audio: np.ndarray) -> List[SpeechSegment]:
        """
        检测音频中的语音片段
        
        Args:
            audio: float32 音频数据，采样率应为 16000 Hz
        
        Returns:
            语音片段列表
        """
        self._load_model()
        
        import torch
        
        # 确保音频格式正确
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        
        # 转换为 torch tensor
        audio_tensor = torch.from_numpy(audio)
        
        # 获取语音时间戳
        get_speech_timestamps = self._utils[0]
        
        speech_timestamps = get_speech_timestamps(
            audio_tensor,
            self._model,
            threshold=self.threshold,
            sampling_rate=self.sample_rate,
            min_speech_duration_ms=int(self.min_speech_duration * 1000),
            min_silence_duration_ms=int(self.min_silence_duration * 1000)
        )
        
        # 转换为 SpeechSegment 列表
        segments = []
        for ts in speech_timestamps:
            start_sample = ts['start']
            end_sample = ts['end']
            
            # 添加 padding
            padding_samples = int(self.padding * self.sample_rate)
            start_sample = max(0, start_sample - padding_samples)
            end_sample = min(len(audio), end_sample + padding_samples)
            
            # 计算时间戳
            start_time = start_sample / self.sample_rate
            end_time = end_sample / self.sample_rate
            
            # 提取音频数据
            segment_audio = audio[start_sample:end_sample].copy()
            
            segments.append(SpeechSegment(
                global_start=start_time,
                global_end=end_time,
                audio_data=segment_audio
            ))
        
        return segments
    
    def detect_with_probabilities(
        self, 
        audio: np.ndarray,
        window_size_samples: int = 512
    ) -> Tuple[List[SpeechSegment], np.ndarray]:
        """
        检测语音片段并返回逐帧概率
        
        Args:
            audio: 音频数据
            window_size_samples: VAD 窗口大小
        
        Returns:
            (语音片段列表, 逐帧概率数组)
        """
        self._load_model()
        
        import torch
        
        # 确保音频格式正确
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        
        audio_tensor = torch.from_numpy(audio)
        
        # 获取逐帧概率
        probabilities = []
        self._model.reset_states()
        
        for i in range(0, len(audio), window_size_samples):
            chunk = audio_tensor[i:i + window_size_samples]
            if len(chunk) < window_size_samples:
                # 填充最后一个块
                chunk = torch.nn.functional.pad(
                    chunk, 
                    (0, window_size_samples - len(chunk))
                )
            
            prob = self._model(chunk, self.sample_rate).item()
            probabilities.append(prob)
        
        probabilities = np.array(probabilities)
        
        # 同时获取片段
        segments = self.detect(audio)
        
        return segments, probabilities


def merge_segments(
    segments: List[SpeechSegment],
    max_gap: float = 0.5,
    audio: Optional[np.ndarray] = None,
    sample_rate: int = TARGET_SAMPLE_RATE
) -> List[SpeechSegment]:
    """
    合并相邻的语音片段
    
    Args:
        segments: 语音片段列表
        max_gap: 最大间隔时长 (秒)，小于此值的间隔将被合并
        audio: 原始音频数据 (用于重新提取合并后的音频)
        sample_rate: 采样率
    
    Returns:
        合并后的片段列表
    """
    if not segments:
        return []
    
    # 按开始时间排序
    sorted_segments = sorted(segments, key=lambda s: s.global_start)
    
    merged = []
    current = sorted_segments[0]
    
    for next_seg in sorted_segments[1:]:
        gap = next_seg.global_start - current.global_end
        
        if gap <= max_gap:
            # 合并片段
            if audio is not None:
                # 重新提取合并后的音频
                start_sample = int(current.global_start * sample_rate)
                end_sample = int(next_seg.global_end * sample_rate)
                merged_audio = audio[start_sample:end_sample].copy()
            else:
                # 简单拼接
                merged_audio = np.concatenate([
                    current.audio_data,
                    next_seg.audio_data
                ])
            
            current = SpeechSegment(
                global_start=current.global_start,
                global_end=next_seg.global_end,
                audio_data=merged_audio
            )
        else:
            merged.append(current)
            current = next_seg
    
    merged.append(current)
    return merged


def filter_short_segments(
    segments: List[SpeechSegment],
    min_duration: float = 0.5
) -> List[SpeechSegment]:
    """
    过滤过短的语音片段
    
    Args:
        segments: 语音片段列表
        min_duration: 最小时长 (秒)
    
    Returns:
        过滤后的片段列表
    """
    return [s for s in segments if s.duration >= min_duration]

