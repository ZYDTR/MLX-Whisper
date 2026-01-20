"""核心数据模型定义"""

from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np


@dataclass
class SpeechSegment:
    """VAD 检测出的语音片段"""
    global_start: float    # 相对原始音频的开始时间（秒）
    global_end: float      # 相对原始音频的结束时间（秒）
    audio_data: np.ndarray # 该片段的音频数据 (float32)
    
    def to_global_timestamp(self, local_ts: float) -> float:
        """将片段内局部时间戳转换为全局时间戳"""
        return self.global_start + local_ts
    
    @property
    def duration(self) -> float:
        """片段时长（秒）"""
        return self.global_end - self.global_start


@dataclass
class Word:
    """词级转录结果"""
    text: str
    start: float  # 全局时间戳
    end: float    # 全局时间戳
    speaker: Optional[str] = None
    
    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class TranscriptionSegment:
    """转录片段，包含说话人信息"""
    start: float
    end: float
    text: str
    speaker: str
    words: List[Word] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "speaker": self.speaker,
            "text": self.text
        }


@dataclass
class SpeakerInfo:
    """说话人信息"""
    id: str
    total_duration: float = 0.0
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "total_duration": round(self.total_duration, 3)
        }


@dataclass
class ProcessingResult:
    """完整处理结果"""
    audio_file: str
    duration: float
    sample_rate: int
    num_speakers: int
    processing_time: float
    speakers: List[SpeakerInfo]
    segments: List[TranscriptionSegment]
    
    def to_dict(self) -> dict:
        return {
            "metadata": {
                "audio_file": self.audio_file,
                "duration": round(self.duration, 3),
                "sample_rate": self.sample_rate,
                "num_speakers": self.num_speakers,
                "processing_time": round(self.processing_time, 3)
            },
            "speakers": [s.to_dict() for s in self.speakers],
            "segments": [seg.to_dict() for seg in self.segments]
        }



