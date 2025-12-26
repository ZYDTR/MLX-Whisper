"""主流水线模块

整合所有处理阶段，实现完整的语音处理流程
支持 ASR 和声纹分割的并行执行
支持带上下文的 ASR 转录，提升跨片段语义连贯性
"""

import time
import json
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple, Dict
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed

from .models import (
    ProcessingResult, TranscriptionSegment, 
    SpeakerInfo, Word
)
from .preprocessor import (
    load_audio, get_audio_info, chunk_audio,
    TARGET_SAMPLE_RATE
)
from .vad import VADDetector, merge_segments, filter_short_segments
from .asr import ASRTranscriber
from .diarization import SpeakerDiarizer, merge_speaker_segments
from .alignment import align_transcription_with_speakers


@dataclass
class TimingStats:
    """时间统计"""
    total_time: float = 0.0
    audio_loading: float = 0.0
    vad_detection: float = 0.0
    vad_merging: float = 0.0
    asr_transcription: float = 0.0
    speaker_diarization: float = 0.0
    alignment: float = 0.0
    chunk_processing: float = 0.0
    other: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        """转换为字典"""
        return {
            "total_time": self.total_time,
            "audio_loading": self.audio_loading,
            "vad_detection": self.vad_detection,
            "vad_merging": self.vad_merging,
            "asr_transcription": self.asr_transcription,
            "speaker_diarization": self.speaker_diarization,
            "alignment": self.alignment,
            "chunk_processing": self.chunk_processing,
            "other": self.other
        }
    
    def print_summary(self):
        """打印时间统计摘要"""
        print("\n" + "=" * 70)
        print("⏱️  详细时间统计")
        print("=" * 70)
        
        if self.total_time == 0:
            print("⚠️  未记录时间统计")
            return
        
        stats = [
            ("总处理时间", self.total_time, 100.0),
            ("音频加载与预处理", self.audio_loading, self.audio_loading / self.total_time * 100),
            ("VAD 检测", self.vad_detection, self.vad_detection / self.total_time * 100),
            ("VAD 片段合并", self.vad_merging, self.vad_merging / self.total_time * 100),
            ("Whisper ASR 转录", self.asr_transcription, self.asr_transcription / self.total_time * 100),
            ("声纹分割 (Diarization)", self.speaker_diarization, self.speaker_diarization / self.total_time * 100),
            ("时间对齐", self.alignment, self.alignment / self.total_time * 100),
            ("分块处理开销", self.chunk_processing, self.chunk_processing / self.total_time * 100),
            ("其他", self.other, self.other / self.total_time * 100),
        ]
        
        print(f"{'阶段':<25} {'耗时(秒)':<15} {'占比':<10}")
        print("-" * 70)
        for name, duration, percentage in stats:
            print(f"{name:<25} {duration:<15.2f} {percentage:<10.2f}%")
        
        print("=" * 70)


@dataclass
class PipelineConfig:
    """流水线配置"""
    # ASR 配置
    whisper_model: str = "large-v3"
    language: Optional[str] = None
    batch_size: int = 12
    quantization: Optional[str] = None
    context_duration: float = 3.0  # ASR 上下文时长（秒）
    
    # VAD 配置
    vad_threshold: float = 0.5
    min_speech_duration: float = 0.25
    min_silence_duration: float = 0.1
    vad_merge_gap: float = 0.5  # VAD 片段合并间隔
    
    # 声纹分割配置
    num_speakers: Optional[int] = None
    min_speakers: int = 1
    max_speakers: int = 10
    
    # 对齐配置
    alignment_method: str = "iou"
    merge_gap: float = 2.0  # 增大合并间隔，提升连贯性
    
    # 长音频分块配置
    chunk_duration: float = 600.0  # 10 分钟
    overlap_duration: float = 30.0  # 30 秒重叠
    long_audio_threshold: float = 1800.0  # 30 分钟以上启用分块
    
    # 并行配置
    parallel_processing: bool = True


class SpeechPipeline:
    """语音处理主流水线"""
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        """
        初始化流水线
        
        Args:
            config: 流水线配置，None 使用默认配置
        """
        self.config = config or PipelineConfig()
        
        # 懒加载各模块
        self._vad = None
        self._asr = None
        self._diarizer = None
        
        # 时间统计
        self.timing_stats = TimingStats()
    
    @property
    def vad(self) -> VADDetector:
        if self._vad is None:
            self._vad = VADDetector(
                threshold=self.config.vad_threshold,
                min_speech_duration=self.config.min_speech_duration,
                min_silence_duration=self.config.min_silence_duration
            )
        return self._vad
    
    @property
    def asr(self) -> ASRTranscriber:
        if self._asr is None:
            self._asr = ASRTranscriber(
                model_name=self.config.whisper_model,
                language=self.config.language,
                batch_size=self.config.batch_size,
                quantization=self.config.quantization,
                context_duration=self.config.context_duration
            )
        return self._asr
    
    @property
    def diarizer(self) -> SpeakerDiarizer:
        if self._diarizer is None:
            self._diarizer = SpeakerDiarizer(
                num_speakers=self.config.num_speakers,
                min_speakers=self.config.min_speakers,
                max_speakers=self.config.max_speakers
            )
        return self._diarizer
    
    def _process_asr_branch(
        self, 
        audio: np.ndarray,
        full_audio: Optional[np.ndarray] = None,
        record_timing: bool = True
    ) -> Tuple[List[Word], Dict[str, float]]:
        """
        ASR 分支处理（支持上下文）
        
        Args:
            audio: 当前处理的音频（用于 VAD）
            full_audio: 完整音频（用于 ASR 上下文，None 则使用 audio）
            record_timing: 是否记录时间
        
        Returns:
            (词列表, 时间统计字典)
        """
        timing = {"vad": 0.0, "vad_merge": 0.0, "asr": 0.0}
        
        # 使用完整音频作为上下文来源
        context_audio = full_audio if full_audio is not None else audio
        
        # VAD 检测
        t0 = time.time()
        segments = self.vad.detect(audio)
        timing["vad"] = time.time() - t0
        
        # 合并相邻片段
        t0 = time.time()
        segments = merge_segments(
            segments, 
            max_gap=self.config.vad_merge_gap, 
            audio=audio
        )
        timing["vad_merge"] = time.time() - t0
        
        # 过滤过短片段
        segments = filter_short_segments(segments, min_duration=0.5)
        
        # 转录（传入完整音频用于上下文）
        t0 = time.time()
        words = self.asr.transcribe_segments(
            segments, 
            full_audio=context_audio
        )
        timing["asr"] = time.time() - t0
        
        return words, timing
    
    def _process_sd_branch(
        self, 
        audio: np.ndarray
    ) -> Tuple[List[Tuple[str, float, float]], float]:
        """
        声纹分割分支处理
        
        Returns:
            (说话人片段列表, 耗时)
        """
        t0 = time.time()
        segments = self.diarizer.diarize(audio)
        
        # 合并相邻的同说话人片段
        segments = merge_speaker_segments(segments, max_gap=0.5)
        elapsed = time.time() - t0
        
        return segments, elapsed
    
    def process(
        self, 
        audio_path: str,
        output_path: Optional[str] = None
    ) -> ProcessingResult:
        """
        处理音频文件
        
        Args:
            audio_path: 输入音频文件路径
            output_path: 输出 JSON 文件路径 (可选)
        
        Returns:
            处理结果
        """
        start_time = time.time()
        
        # 获取音频信息
        audio_info = get_audio_info(audio_path)
        print(f"📁 加载音频: {audio_info['file']}")
        print(f"   时长: {audio_info['duration']:.1f}s, "
              f"采样率: {audio_info['sample_rate']}Hz, "
              f"声道: {audio_info['channels']}")
        
        # 加载并预处理音频
        print("🔄 预处理音频...")
        t0 = time.time()
        audio, sr = load_audio(audio_path)
        self.timing_stats.audio_loading = time.time() - t0
        duration = len(audio) / sr
        
        # 判断是否需要分块处理
        if duration > self.config.long_audio_threshold:
            result = self._process_long_audio(audio, sr, audio_info)
        else:
            result = self._process_short_audio(audio, sr, audio_info)
        
        # 计算处理时间
        processing_time = time.time() - start_time
        result.processing_time = processing_time
        self.timing_stats.total_time = processing_time
        
        # 计算其他时间（总时间减去已记录的时间）
        # 注意：chunk_processing 只记录分块循环、去重等开销，不包含块处理时间
        recorded_time = (
            self.timing_stats.audio_loading +
            self.timing_stats.vad_detection +
            self.timing_stats.vad_merging +
            self.timing_stats.asr_transcription +
            self.timing_stats.speaker_diarization +
            self.timing_stats.alignment +
            self.timing_stats.chunk_processing
        )
        self.timing_stats.other = max(0, processing_time - recorded_time)
        
        print(f"\n✅ 处理完成!")
        print(f"   处理时间: {processing_time:.1f}s")
        print(f"   实时比: {duration / processing_time:.1f}x")
        print(f"   识别说话人数: {result.num_speakers}")
        print(f"   转录片段数: {len(result.segments)}")
        
        # 打印时间统计
        self.timing_stats.print_summary()
        
        # 保存结果
        if output_path:
            self._save_result(result, output_path)
            print(f"\n   输出文件: {output_path}")
        
        return result
    
    def _process_short_audio(
        self,
        audio: np.ndarray,
        sample_rate: int,
        audio_info: dict
    ) -> ProcessingResult:
        """处理短音频（不分块）"""
        
        if self.config.parallel_processing:
            # 并行处理 ASR 和声纹分割
            print("🚀 并行处理 ASR 和声纹分割（带上下文）...")
            
            with ThreadPoolExecutor(max_workers=2) as executor:
                # ASR 分支使用完整音频作为上下文
                asr_future = executor.submit(
                    self._process_asr_branch, audio, audio
                )
                sd_future = executor.submit(self._process_sd_branch, audio)
                
                words, asr_timing = asr_future.result()
                speaker_segments, sd_time = sd_future.result()
                
                # 记录时间
                self.timing_stats.vad_detection += asr_timing["vad"]
                self.timing_stats.vad_merging += asr_timing["vad_merge"]
                self.timing_stats.asr_transcription += asr_timing["asr"]
                self.timing_stats.speaker_diarization += sd_time
        else:
            # 串行处理
            print("🎤 VAD + ASR 处理（带上下文）...")
            words, asr_timing = self._process_asr_branch(audio, audio)
            
            print("👥 声纹分割处理...")
            speaker_segments, sd_time = self._process_sd_branch(audio)
            
            # 记录时间
            self.timing_stats.vad_detection += asr_timing["vad"]
            self.timing_stats.vad_merging += asr_timing["vad_merge"]
            self.timing_stats.asr_transcription += asr_timing["asr"]
            self.timing_stats.speaker_diarization += sd_time
        
        # 对齐
        print("🔗 时间对齐...")
        t0 = time.time()
        segments, speaker_stats = align_transcription_with_speakers(
            words, speaker_segments,
            merge_gap=self.config.merge_gap,
            alignment_method=self.config.alignment_method
        )
        self.timing_stats.alignment = time.time() - t0
        
        return ProcessingResult(
            audio_file=audio_info['file'],
            duration=audio_info['duration'],
            sample_rate=sample_rate,
            num_speakers=len(speaker_stats),
            processing_time=0,  # 稍后填充
            speakers=speaker_stats,
            segments=segments
        )
    
    def _process_long_audio(
        self,
        audio: np.ndarray,
        sample_rate: int,
        audio_info: dict
    ) -> ProcessingResult:
        """处理长音频（分块处理，带上下文）"""
        print(f"📦 长音频分块处理 (时长: {len(audio)/sample_rate:.0f}s)...")
        print(f"   🔗 ASR 上下文: ±{self.config.context_duration}s")
        
        # 分块
        chunks = chunk_audio(
            audio, sample_rate,
            chunk_duration=self.config.chunk_duration,
            overlap_duration=self.config.overlap_duration
        )
        
        print(f"   分成 {len(chunks)} 块处理")
        
        all_words = []
        all_speaker_segments = []
        
        for i, (chunk_audio_data, chunk_start, chunk_end) in enumerate(chunks):
            print(f"\n   处理块 {i+1}/{len(chunks)} "
                  f"[{chunk_start:.0f}s - {chunk_end:.0f}s]")
            
            chunk_t0 = time.time()
            
            # 为当前块扩展上下文范围（从完整音频中提取）
            context_margin = self.config.context_duration
            context_start_sample = max(0, int((chunk_start - context_margin) * sample_rate))
            context_end_sample = min(len(audio), int((chunk_end + context_margin) * sample_rate))
            chunk_with_context = audio[context_start_sample:context_end_sample]
            
            # 处理当前块
            if self.config.parallel_processing:
                with ThreadPoolExecutor(max_workers=2) as executor:
                    # ASR 使用带上下文的音频块
                    asr_future = executor.submit(
                        self._process_asr_branch_for_chunk,
                        chunk_audio_data,
                        chunk_with_context,
                        chunk_start,
                        context_start_sample / sample_rate
                    )
                    sd_future = executor.submit(
                        self._process_sd_branch, chunk_audio_data
                    )
                    
                    chunk_words, asr_timing = asr_future.result()
                    chunk_speaker_segs, sd_time = sd_future.result()
                    
                    # 记录时间
                    self.timing_stats.vad_detection += asr_timing["vad"]
                    self.timing_stats.vad_merging += asr_timing["vad_merge"]
                    self.timing_stats.asr_transcription += asr_timing["asr"]
                    self.timing_stats.speaker_diarization += sd_time
            else:
                chunk_words, asr_timing = self._process_asr_branch_for_chunk(
                    chunk_audio_data,
                    chunk_with_context,
                    chunk_start,
                    context_start_sample / sample_rate
                )
                chunk_speaker_segs, sd_time = self._process_sd_branch(chunk_audio_data)
                
                # 记录时间
                self.timing_stats.vad_detection += asr_timing["vad"]
                self.timing_stats.vad_merging += asr_timing["vad_merge"]
                self.timing_stats.asr_transcription += asr_timing["asr"]
                self.timing_stats.speaker_diarization += sd_time
            
            chunk_elapsed = time.time() - chunk_t0
            print(f"      块处理耗时: {chunk_elapsed:.1f}s")
            
            # 调整时间戳（加上块偏移）
            t0 = time.time()
            for word in chunk_words:
                word.start += chunk_start
                word.end += chunk_start
            
            chunk_speaker_segs = [
                (speaker, start + chunk_start, end + chunk_start)
                for speaker, start, end in chunk_speaker_segs
            ]
            
            # 去重（重叠区域优先取后块结果）
            if i > 0:
                overlap_start = chunk_start
                # 移除前一块在重叠区的结果
                all_words = [w for w in all_words if w.end < overlap_start]
                all_speaker_segments = [
                    seg for seg in all_speaker_segments 
                    if seg[2] < overlap_start
                ]
            
            all_words.extend(chunk_words)
            all_speaker_segments.extend(chunk_speaker_segs)
            self.timing_stats.chunk_processing += time.time() - t0
        
        # 对齐
        print("\n🔗 全局时间对齐...")
        t0 = time.time()
        segments, speaker_stats = align_transcription_with_speakers(
            all_words, all_speaker_segments,
            merge_gap=self.config.merge_gap,
            alignment_method=self.config.alignment_method
        )
        self.timing_stats.alignment = time.time() - t0
        
        return ProcessingResult(
            audio_file=audio_info['file'],
            duration=audio_info['duration'],
            sample_rate=sample_rate,
            num_speakers=len(speaker_stats),
            processing_time=0,
            speakers=speaker_stats,
            segments=segments
        )
    
    def _process_asr_branch_for_chunk(
        self,
        chunk_audio: np.ndarray,
        chunk_with_context: np.ndarray,
        chunk_global_start: float,
        context_global_start: float
    ) -> Tuple[List[Word], Dict[str, float]]:
        """
        处理单个音频块的 ASR（带上下文）
        
        Args:
            chunk_audio: 原始音频块（用于 VAD）
            chunk_with_context: 带上下文的音频块（用于 ASR）
            chunk_global_start: 块在全局的开始时间
            context_global_start: 带上下文的块在全局的开始时间
        """
        timing = {"vad": 0.0, "vad_merge": 0.0, "asr": 0.0}
        
        # VAD 检测（在原始块上）
        t0 = time.time()
        segments = self.vad.detect(chunk_audio)
        timing["vad"] = time.time() - t0
        
        # 合并相邻片段
        t0 = time.time()
        segments = merge_segments(
            segments, 
            max_gap=self.config.vad_merge_gap, 
            audio=chunk_audio
        )
        timing["vad_merge"] = time.time() - t0
        
        # 过滤过短片段
        segments = filter_short_segments(segments, min_duration=0.5)
        
        # 调整片段时间戳以匹配带上下文的音频
        context_offset = chunk_global_start - context_global_start
        adjusted_segments = []
        for seg in segments:
            adjusted_seg = type(seg)(
                global_start=seg.global_start + context_offset,
                global_end=seg.global_end + context_offset,
                audio_data=seg.audio_data
            )
            adjusted_segments.append(adjusted_seg)
        
        # 转录（使用带上下文的音频）
        t0 = time.time()
        words = self.asr.transcribe_segments(
            adjusted_segments, 
            full_audio=chunk_with_context
        )
        timing["asr"] = time.time() - t0
        
        # 将词时间戳调整回块内相对位置
        for word in words:
            word.start -= context_offset
            word.end -= context_offset
        
        return words, timing
    
    def _save_result(self, result: ProcessingResult, output_path: str):
        """保存处理结果为 JSON"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)


def process_audio(
    audio_path: str,
    output_path: Optional[str] = None,
    whisper_model: str = "large-v3",
    language: Optional[str] = None,
    num_speakers: Optional[int] = None,
    context_duration: float = 3.0,
    **kwargs
) -> ProcessingResult:
    """
    便捷函数：处理音频文件
    
    Args:
        audio_path: 输入音频文件路径
        output_path: 输出 JSON 文件路径 (可选)
        whisper_model: Whisper 模型名称
        language: 语言代码 (None 自动检测)
        num_speakers: 说话人数量 (None 自动检测)
        context_duration: ASR 上下文时长（秒），默认 3 秒
        **kwargs: 其他配置参数
    
    Returns:
        处理结果
    """
    config = PipelineConfig(
        whisper_model=whisper_model,
        language=language,
        num_speakers=num_speakers,
        context_duration=context_duration,
        **kwargs
    )
    
    pipeline = SpeechPipeline(config)
    return pipeline.process(audio_path, output_path)
