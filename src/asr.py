"""阶段 2: ASR 语音转文字模块

使用 mlx-whisper 进行语音转录
- 利用 MLX 框架深度优化 Apple Silicon GPU
- 支持批量推理实现超实时转录
- 支持前后上下文，提升跨片段语义连贯性
- 输出词级时间戳
"""

import numpy as np
from typing import List, Optional, Dict, Any, Tuple
from .models import SpeechSegment, Word
from .preprocessor import TARGET_SAMPLE_RATE


class ASRTranscriber:
    """mlx-whisper 语音转录器"""
    
    # 可用模型列表
    AVAILABLE_MODELS = [
        "tiny", "tiny.en",
        "base", "base.en", 
        "small", "small.en",
        "medium", "medium.en",
        "large", "large-v2", "large-v3",
        "distil-large-v2", "distil-large-v3",
        "turbo", "large-v3-turbo"
    ]
    
    # 模型路径映射（特殊模型使用不同路径格式）
    MODEL_PATH_MAP = {
        "turbo": "mlx-community/whisper-large-v3-turbo",
        "large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
    }
    
    def __init__(
        self,
        model_name: str = "large-v3",
        language: Optional[str] = None,
        batch_size: int = 12,
        quantization: Optional[str] = None,  # 已禁用，保留参数兼容性
        context_duration: float = 3.0  # 前后上下文时长（秒）
    ):
        """
        初始化 ASR 转录器
        
        Args:
            model_name: Whisper 模型名称
            language: 语言代码 (None 表示自动检测)
            batch_size: 批处理大小
            quantization: 量化选项 (暂未使用)
            context_duration: 转录时包含的前后上下文时长（秒）
        """
        self.model_name = model_name
        self.language = language
        self.batch_size = batch_size
        self.quantization = quantization
        self.context_duration = context_duration
        
        self._loaded = False
    
    def _get_model_path(self) -> str:
        """获取模型的 HuggingFace 路径"""
        # 检查是否有特殊路径映射
        if self.model_name in self.MODEL_PATH_MAP:
            return self.MODEL_PATH_MAP[self.model_name]
        # 默认路径格式
        return f"mlx-community/whisper-{self.model_name}-mlx"
    
    def _ensure_loaded(self):
        """确保模型已加载 (mlx-whisper 会自动加载)"""
        self._loaded = True
    
    def _extract_context_audio(
        self,
        segment: SpeechSegment,
        full_audio: np.ndarray,
        sample_rate: int = TARGET_SAMPLE_RATE
    ) -> Tuple[np.ndarray, float, float]:
        """
        提取带上下文的音频片段
        
        Args:
            segment: 原始语音片段
            full_audio: 完整音频数据
            sample_rate: 采样率
        
        Returns:
            (带上下文的音频, 上下文音频的全局开始时间, 原片段在上下文中的相对开始时间)
        """
        total_duration = len(full_audio) / sample_rate
        
        # 计算上下文边界
        context_start = max(0, segment.global_start - self.context_duration)
        context_end = min(total_duration, segment.global_end + self.context_duration)
        
        # 转换为采样点
        start_sample = int(context_start * sample_rate)
        end_sample = int(context_end * sample_rate)
        
        # 提取带上下文的音频
        context_audio = full_audio[start_sample:end_sample].copy()
        
        # 原片段在上下文音频中的相对位置
        relative_start = segment.global_start - context_start
        
        return context_audio, context_start, relative_start
    
    def transcribe_segment(
        self,
        segment: SpeechSegment,
        task: str = "transcribe",
        full_audio: Optional[np.ndarray] = None
    ) -> List[Word]:
        """
        转录单个语音片段（支持上下文）
        
        Args:
            segment: 语音片段
            task: 任务类型 ("transcribe" 或 "translate")
            full_audio: 完整音频数据（用于提取上下文，None 则不使用上下文）
        
        Returns:
            词级转录结果列表 (全局时间戳)
        """
        self._ensure_loaded()
        
        import mlx_whisper
        
        # 决定使用的音频和时间偏移
        if full_audio is not None and self.context_duration > 0:
            # 使用带上下文的音频
            context_audio, context_global_start, relative_segment_start = \
                self._extract_context_audio(segment, full_audio)
            audio_to_transcribe = context_audio
            
            # 原片段在上下文中的时间边界（用于过滤）
            segment_start_in_context = relative_segment_start
            segment_end_in_context = relative_segment_start + segment.duration
        else:
            # 不使用上下文，直接使用片段音频
            audio_to_transcribe = segment.audio_data
            context_global_start = segment.global_start
            segment_start_in_context = 0
            segment_end_in_context = segment.duration
        
        # 获取模型路径
        model_path = self._get_model_path()
        
        # 执行转录
        result = mlx_whisper.transcribe(
            audio_to_transcribe,
            path_or_hf_repo=model_path,
            language=self.language,
            task=task,
            word_timestamps=True,
            verbose=False
        )
        
        # 转换为 Word 列表，只保留原片段时间范围内的词
        words = []
        
        if "segments" in result:
            for seg in result["segments"]:
                if "words" in seg:
                    for word_info in seg["words"]:
                        local_start = word_info.get("start", 0)
                        local_end = word_info.get("end", 0)
                        
                        # 检查词是否在原片段的时间范围内
                        # 使用词的中心点判断归属
                        word_center = (local_start + local_end) / 2
                        
                        # 添加少量容差，确保边界词不丢失
                        tolerance = 0.1
                        if (segment_start_in_context - tolerance <= word_center <= 
                            segment_end_in_context + tolerance):
                            
                            # 计算全局时间戳
                            global_start = context_global_start + local_start
                            global_end = context_global_start + local_end
                            
                            words.append(Word(
                                text=word_info.get("word", "").strip(),
                                start=global_start,
                                end=global_end
                            ))
        
        return words
    
    def transcribe_segments(
        self,
        segments: List[SpeechSegment],
        task: str = "transcribe",
        full_audio: Optional[np.ndarray] = None
    ) -> List[Word]:
        """
        批量转录多个语音片段（支持上下文）
        
        Args:
            segments: 语音片段列表
            task: 任务类型
            full_audio: 完整音频数据（用于提取上下文）
        
        Returns:
            所有词级转录结果 (全局时间戳，按时间排序)
        """
        all_words = []
        
        for segment in segments:
            words = self.transcribe_segment(segment, task, full_audio)
            all_words.extend(words)
        
        # 按开始时间排序
        all_words.sort(key=lambda w: w.start)
        
        # 去重（上下文可能导致边界词重复）
        all_words = self._deduplicate_words(all_words)
        
        return all_words
    
    def _deduplicate_words(
        self,
        words: List[Word],
        time_tolerance: float = 0.1,
        text_match: bool = True
    ) -> List[Word]:
        """
        去除重复的词
        
        Args:
            words: 词列表（已按时间排序）
            time_tolerance: 时间容差（秒）
            text_match: 是否要求文本也匹配
        
        Returns:
            去重后的词列表
        """
        if not words:
            return []
        
        deduplicated = [words[0]]
        
        for word in words[1:]:
            prev_word = deduplicated[-1]
            
            # 检查是否重复
            time_overlap = abs(word.start - prev_word.start) < time_tolerance
            
            if text_match:
                is_duplicate = time_overlap and word.text == prev_word.text
            else:
                is_duplicate = time_overlap
            
            if not is_duplicate:
                deduplicated.append(word)
        
        return deduplicated
    
    def transcribe_audio(
        self,
        audio: np.ndarray,
        task: str = "transcribe"
    ) -> Dict[str, Any]:
        """
        直接转录完整音频 (不经过 VAD)
        
        Args:
            audio: 音频数据
            task: 任务类型
        
        Returns:
            原始转录结果字典
        """
        self._ensure_loaded()
        
        import mlx_whisper
        
        return mlx_whisper.transcribe(
            audio,
            path_or_hf_repo=self._get_model_path(),
            language=self.language,
            task=task,
            word_timestamps=True,
            verbose=False
        )


def words_to_text(words: List[Word]) -> str:
    """
    将词列表转换为连续文本
    
    Args:
        words: 词列表
    
    Returns:
        连续文本
    """
    return " ".join(w.text for w in words if w.text)


def group_words_by_gap(
    words: List[Word],
    max_gap: float = 1.0
) -> List[List[Word]]:
    """
    根据时间间隔将词分组
    
    Args:
        words: 词列表
        max_gap: 最大间隔时长 (秒)
    
    Returns:
        词组列表
    """
    if not words:
        return []
    
    groups = []
    current_group = [words[0]]
    
    for word in words[1:]:
        gap = word.start - current_group[-1].end
        
        if gap > max_gap:
            groups.append(current_group)
            current_group = [word]
        else:
            current_group.append(word)
    
    if current_group:
        groups.append(current_group)
    
    return groups
