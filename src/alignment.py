"""阶段 4: 时间对齐模块

使用 IoU (Intersection over Union) 算法将 ASR 词级时间戳
与声纹分割结果进行精准对齐
"""

from typing import List, Tuple, Optional
from .models import Word, TranscriptionSegment, SpeakerInfo


def calculate_iou(
    start1: float, end1: float,
    start2: float, end2: float
) -> float:
    """
    计算两个时间段的 IoU (Intersection over Union)
    
    Args:
        start1, end1: 第一个时间段
        start2, end2: 第二个时间段
    
    Returns:
        IoU 值 (0-1)
    """
    # 计算交集
    intersection_start = max(start1, start2)
    intersection_end = min(end1, end2)
    intersection = max(0, intersection_end - intersection_start)
    
    # 计算并集
    union = (end1 - start1) + (end2 - start2) - intersection
    
    # 计算 IoU
    return intersection / union if union > 0 else 0


def calculate_overlap_ratio(
    start1: float, end1: float,
    start2: float, end2: float
) -> float:
    """
    计算第一个时间段被第二个时间段覆盖的比例
    
    Args:
        start1, end1: 第一个时间段 (被覆盖)
        start2, end2: 第二个时间段 (覆盖者)
    
    Returns:
        覆盖比例 (0-1)
    """
    # 计算交集
    intersection_start = max(start1, start2)
    intersection_end = min(end1, end2)
    intersection = max(0, intersection_end - intersection_start)
    
    # 计算第一个时间段的长度
    duration1 = end1 - start1
    
    return intersection / duration1 if duration1 > 0 else 0


def assign_speaker(
    word_start: float,
    word_end: float,
    speaker_segments: List[Tuple[str, float, float]],
    method: str = "iou"
) -> str:
    """
    为单个词分配说话人 ID
    
    Args:
        word_start: 词开始时间
        word_end: 词结束时间
        speaker_segments: 说话人时间段列表 [(speaker_id, start, end), ...]
        method: 匹配方法 ("iou" 或 "overlap")
    
    Returns:
        说话人 ID，若无匹配则返回 "UNKNOWN"
    """
    best_speaker = "UNKNOWN"
    best_score = 0.0
    
    for speaker_id, seg_start, seg_end in speaker_segments:
        if method == "iou":
            score = calculate_iou(word_start, word_end, seg_start, seg_end)
        else:  # overlap
            score = calculate_overlap_ratio(word_start, word_end, seg_start, seg_end)
        
        if score > best_score:
            best_score = score
            best_speaker = speaker_id
    
    return best_speaker


def assign_speakers_to_words(
    words: List[Word],
    speaker_segments: List[Tuple[str, float, float]],
    method: str = "iou"
) -> List[Word]:
    """
    为所有词分配说话人
    
    Args:
        words: 词列表
        speaker_segments: 说话人时间段列表
        method: 匹配方法
    
    Returns:
        带说话人标签的词列表
    """
    for word in words:
        word.speaker = assign_speaker(
            word.start, word.end,
            speaker_segments,
            method
        )
    
    return words


def group_words_by_speaker(
    words: List[Word],
    max_gap: float = 1.0
) -> List[TranscriptionSegment]:
    """
    将词按说话人和时间间隔分组为转录片段
    
    Args:
        words: 带说话人标签的词列表
        max_gap: 最大间隔时长 (秒)，超过此值将开始新片段
    
    Returns:
        转录片段列表
    """
    if not words:
        return []
    
    segments = []
    current_words = [words[0]]
    current_speaker = words[0].speaker
    
    for word in words[1:]:
        # 检查是否需要开始新片段
        time_gap = word.start - current_words[-1].end
        speaker_changed = word.speaker != current_speaker
        
        if speaker_changed or time_gap > max_gap:
            # 创建新片段
            segments.append(TranscriptionSegment(
                start=current_words[0].start,
                end=current_words[-1].end,
                speaker=current_speaker or "UNKNOWN",
                text=" ".join(w.text for w in current_words if w.text),
                words=current_words.copy()
            ))
            current_words = [word]
            current_speaker = word.speaker
        else:
            current_words.append(word)
    
    # 添加最后一个片段
    if current_words:
        segments.append(TranscriptionSegment(
            start=current_words[0].start,
            end=current_words[-1].end,
            speaker=current_speaker or "UNKNOWN",
            text=" ".join(w.text for w in current_words if w.text),
            words=current_words.copy()
        ))
    
    return segments


def calculate_speaker_stats(
    segments: List[TranscriptionSegment]
) -> List[SpeakerInfo]:
    """
    计算说话人统计信息
    
    Args:
        segments: 转录片段列表
    
    Returns:
        说话人信息列表
    """
    speaker_durations = {}
    
    for segment in segments:
        speaker = segment.speaker
        duration = segment.end - segment.start
        
        if speaker not in speaker_durations:
            speaker_durations[speaker] = 0.0
        speaker_durations[speaker] += duration
    
    return [
        SpeakerInfo(id=speaker_id, total_duration=duration)
        for speaker_id, duration in sorted(speaker_durations.items())
    ]


def merge_adjacent_segments(
    segments: List[TranscriptionSegment],
    max_gap: float = 0.3
) -> List[TranscriptionSegment]:
    """
    合并相同说话人的相邻片段
    
    Args:
        segments: 转录片段列表
        max_gap: 最大间隔时长 (秒)
    
    Returns:
        合并后的片段列表
    """
    if not segments:
        return []
    
    merged = []
    current = segments[0]
    
    for next_seg in segments[1:]:
        gap = next_seg.start - current.end
        
        if next_seg.speaker == current.speaker and gap <= max_gap:
            # 合并片段
            merged_text = current.text + " " + next_seg.text
            merged_words = current.words + next_seg.words
            
            current = TranscriptionSegment(
                start=current.start,
                end=next_seg.end,
                speaker=current.speaker,
                text=merged_text.strip(),
                words=merged_words
            )
        else:
            merged.append(current)
            current = next_seg
    
    merged.append(current)
    
    return merged


def align_transcription_with_speakers(
    words: List[Word],
    speaker_segments: List[Tuple[str, float, float]],
    merge_gap: float = 1.0,
    alignment_method: str = "iou"
) -> Tuple[List[TranscriptionSegment], List[SpeakerInfo]]:
    """
    完整的对齐流程：将转录词与说话人分割结果对齐
    
    Args:
        words: 词级转录结果
        speaker_segments: 说话人时间段列表
        merge_gap: 合并片段的最大间隔
        alignment_method: 对齐方法
    
    Returns:
        (转录片段列表, 说话人信息列表)
    """
    # 1. 为每个词分配说话人
    words_with_speakers = assign_speakers_to_words(
        words, speaker_segments, alignment_method
    )
    
    # 2. 按说话人分组
    segments = group_words_by_speaker(words_with_speakers, merge_gap)
    
    # 3. 合并相邻的同说话人片段
    segments = merge_adjacent_segments(segments, max_gap=0.3)
    
    # 4. 计算说话人统计
    speaker_stats = calculate_speaker_stats(segments)
    
    return segments, speaker_stats

