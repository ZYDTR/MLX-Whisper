"""阶段 3: 声纹分割模块 (Speaker Diarization)

使用 pyannote.audio 进行说话人识别与分割
- 业界领先的声纹分割效果
- 支持自动检测说话人数量
"""

import numpy as np
from typing import List, Tuple, Optional
from .preprocessor import TARGET_SAMPLE_RATE


class SpeakerDiarizer:
    """pyannote.audio 声纹分割器"""
    
    def __init__(
        self,
        num_speakers: Optional[int] = None,
        min_speakers: int = 1,
        max_speakers: int = 10,
        min_segment_duration: float = 0.5,
        hf_token: Optional[str] = None
    ):
        """
        初始化声纹分割器
        
        Args:
            num_speakers: 说话人数量 (None 表示自动检测)
            min_speakers: 最小说话人数量 (自动检测时使用)
            max_speakers: 最大说话人数量 (自动检测时使用)
            min_segment_duration: 最小片段时长 (秒)
            hf_token: HuggingFace API token (用于下载模型)
                     如果为 None，会尝试从环境变量 HF_TOKEN 读取
        """
        self.num_speakers = num_speakers
        self.min_speakers = min_speakers
        self.max_speakers = max_speakers
        self.min_segment_duration = min_segment_duration
        
        # 尝试从环境变量获取 token
        if hf_token is None:
            import os
            hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
        
        self.hf_token = hf_token
        
        self._pipeline = None
    
    def _load_model(self):
        """懒加载 pyannote 模型"""
        if self._pipeline is None:
            try:
                import warnings
                # 忽略 torchcodec 警告（我们使用 librosa 加载音频）
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=UserWarning)
                    from pyannote.audio import Pipeline
                    import torch
                
                print("📥 加载 pyannote 声纹分割模型...")
                # 加载预训练模型
                # API: from_pretrained(checkpoint, token=None, ...)
                self._pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1",
                    token=self.hf_token
                )
                
                # 如果有 MPS (Apple Silicon)，使用它
                if torch.backends.mps.is_available():
                    self._pipeline.to(torch.device("mps"))
                    print("   ✅ 使用 Apple Silicon GPU (MPS)")
                else:
                    print("   ⚠️  MPS 不可用，使用 CPU")
                    
            except ImportError as e:
                print(f"⚠️  无法导入 pyannote.audio: {e}")
                print("   请安装: pip install pyannote.audio")
                print("   将使用简化的声纹分割")
                self._pipeline = "fallback"
            except Exception as e:
                error_msg = str(e)
                if "401" in error_msg or "gated" in error_msg.lower() or "token" in error_msg.lower():
                    print("⚠️  pyannote 模型需要 HuggingFace token 才能访问")
                    print("   原因: pyannote/speaker-diarization-3.1 是受限制的模型")
                    print("   解决方案:")
                    print("   1. 访问 https://hf.co/pyannote/speaker-diarization-3.1 接受用户条件")
                    print("   2. 访问 https://hf.co/settings/tokens 创建 token")
                    print("   3. 设置环境变量: export HF_TOKEN=your_token")
                    print("      或在代码中传入: SpeakerDiarizer(hf_token='your_token')")
                    print("   将使用简化的声纹分割")
                else:
                    print(f"⚠️  无法加载 pyannote 模型: {e}")
                    print("   将使用简化的声纹分割")
                self._pipeline = "fallback"
    
    def diarize(
        self,
        audio: np.ndarray,
        sample_rate: int = TARGET_SAMPLE_RATE
    ) -> List[Tuple[str, float, float]]:
        """
        对音频进行说话人分割
        
        Args:
            audio: 原始完整音频 (float32, 16kHz, mono)
            sample_rate: 采样率
        
        Returns:
            说话人时间段列表: [(speaker_id, start, end), ...]
        """
        self._load_model()
        
        # 确保音频格式正确
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        
        if self._pipeline == "fallback":
            # 使用简化的声纹分割（基于能量）
            return self._fallback_diarize(audio, sample_rate)
        
        try:
            import torch
            import torchaudio
            
            # 转换为 torch tensor
            waveform = torch.from_numpy(audio).unsqueeze(0)
            
            # 创建临时音频输入格式
            audio_input = {
                "waveform": waveform,
                "sample_rate": sample_rate
            }
            
            # 执行声纹分割
            if self.num_speakers is not None:
                result = self._pipeline(
                    audio_input,
                    num_speakers=self.num_speakers
                )
            else:
                result = self._pipeline(
                    audio_input,
                    min_speakers=self.min_speakers,
                    max_speakers=self.max_speakers
                )
            
            # pyannote 4.x 返回 DiarizeOutput 对象
            # 需要访问 .speaker_diarization 属性获取 Annotation
            if hasattr(result, 'speaker_diarization'):
                diarization = result.speaker_diarization
            else:
                # 兼容旧版本
                diarization = result
            
            # 转换为标准格式
            segments = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                start = turn.start
                end = turn.end
                speaker_id = f"SPEAKER_{speaker.split('_')[-1]:0>2}"
                
                # 过滤过短的片段
                if (end - start) >= self.min_segment_duration:
                    segments.append((speaker_id, start, end))
            
            # 按开始时间排序
            segments.sort(key=lambda x: x[1])
            
            return segments
            
        except Exception as e:
            print(f"⚠️ pyannote 处理失败: {e}")
            return self._fallback_diarize(audio, sample_rate)
    
    def _fallback_diarize(
        self,
        audio: np.ndarray,
        sample_rate: int
    ) -> List[Tuple[str, float, float]]:
        """
        简化的声纹分割（当 pyannote 不可用时）
        基于能量检测的简单分割，所有语音归为同一说话人
        """
        # 使用简单的能量检测
        frame_length = int(0.025 * sample_rate)  # 25ms
        hop_length = int(0.010 * sample_rate)    # 10ms
        
        # 计算能量
        energy = []
        for i in range(0, len(audio) - frame_length, hop_length):
            frame = audio[i:i + frame_length]
            energy.append(np.sqrt(np.mean(frame ** 2)))
        
        energy = np.array(energy)
        
        # 动态阈值
        threshold = np.mean(energy) * 0.5
        
        # 检测语音区域
        is_speech = energy > threshold
        
        # 转换为时间段
        segments = []
        in_speech = False
        start_frame = 0
        
        for i, speech in enumerate(is_speech):
            if speech and not in_speech:
                start_frame = i
                in_speech = True
            elif not speech and in_speech:
                start_time = start_frame * hop_length / sample_rate
                end_time = i * hop_length / sample_rate
                if (end_time - start_time) >= self.min_segment_duration:
                    segments.append(("SPEAKER_00", start_time, end_time))
                in_speech = False
        
        # 处理最后一个片段
        if in_speech:
            start_time = start_frame * hop_length / sample_rate
            end_time = len(energy) * hop_length / sample_rate
            if (end_time - start_time) >= self.min_segment_duration:
                segments.append(("SPEAKER_00", start_time, end_time))
        
        return segments
    
    def get_speaker_stats(
        self,
        segments: List[Tuple[str, float, float]]
    ) -> dict:
        """
        计算说话人统计信息
        
        Args:
            segments: 说话人时间段列表
        
        Returns:
            统计信息字典
        """
        speaker_durations = {}
        
        for speaker_id, start, end in segments:
            duration = end - start
            if speaker_id not in speaker_durations:
                speaker_durations[speaker_id] = 0.0
            speaker_durations[speaker_id] += duration
        
        total_duration = sum(speaker_durations.values())
        
        return {
            "num_speakers": len(speaker_durations),
            "speakers": {
                speaker_id: {
                    "duration": duration,
                    "percentage": (duration / total_duration * 100) if total_duration > 0 else 0
                }
                for speaker_id, duration in sorted(speaker_durations.items())
            },
            "total_speech_duration": total_duration
        }


def merge_speaker_segments(
    segments: List[Tuple[str, float, float]],
    max_gap: float = 0.5
) -> List[Tuple[str, float, float]]:
    """
    合并同一说话人的相邻片段
    
    Args:
        segments: 说话人时间段列表
        max_gap: 最大间隔时长 (秒)
    
    Returns:
        合并后的片段列表
    """
    if not segments:
        return []
    
    # 按开始时间排序
    sorted_segments = sorted(segments, key=lambda x: x[1])
    
    merged = []
    current_speaker, current_start, current_end = sorted_segments[0]
    
    for speaker_id, start, end in sorted_segments[1:]:
        if speaker_id == current_speaker and (start - current_end) <= max_gap:
            # 合并同一说话人的相邻片段
            current_end = end
        else:
            merged.append((current_speaker, current_start, current_end))
            current_speaker, current_start, current_end = speaker_id, start, end
    
    merged.append((current_speaker, current_start, current_end))
    
    return merged


def smooth_speaker_segments(
    segments: List[Tuple[str, float, float]],
    min_duration: float = 0.3
) -> List[Tuple[str, float, float]]:
    """
    平滑说话人片段，移除过短的转换
    
    当一个说话人的片段过短，且前后都是同一个其他说话人时，
    将该短片段归并到周围的说话人。
    
    Args:
        segments: 说话人时间段列表
        min_duration: 最小片段时长 (秒)
    
    Returns:
        平滑后的片段列表
    """
    if len(segments) < 3:
        return segments
    
    result = [segments[0]]
    
    for i in range(1, len(segments) - 1):
        speaker_id, start, end = segments[i]
        duration = end - start
        
        prev_speaker = result[-1][0]
        next_speaker = segments[i + 1][0]
        
        if duration < min_duration and prev_speaker == next_speaker:
            # 短片段被同一说话人包围，扩展前一个片段
            result[-1] = (prev_speaker, result[-1][1], end)
        else:
            result.append(segments[i])
    
    # 添加最后一个片段
    result.append(segments[-1])
    
    # 再次合并相邻的同说话人片段
    return merge_speaker_segments(result, max_gap=0.0)
