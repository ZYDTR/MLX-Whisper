# Apple Silicon 本地语音处理流水线

## 1. 项目目标

为 Apple Silicon (M1/M2/M3) 平台构建一个高性能、高精度的本地语音处理流水线。该方案旨在实现对长音频文件的快速分析，并输出带有说话人身份的精确转录文本。

### 核心交付指标

| 指标 | 目标 |
|------|------|
| **极致性能** | 实现对音频文件的超实时处理，目标处理速度远快于音频本身时长 |
| **顶级精度** | 语音转录的抗噪能力和准确率需达到业界领先水平 |
| **高精度分割** | 实现可靠的多说话人声纹分割 (Speaker Diarization) |

---

## 2. 边界与范围 (Scope)

### 平台要求

| 项目 | 规范 |
|------|------|
| 操作系统 | macOS 13.0+ |
| 硬件 | 搭载 Apple Silicon (M1/M2/M3) 芯片的设备 |
| 内存 | 建议 8GB+ 统一内存 |

### 输入规范

| 项目 | 规范 |
|------|------|
| 文件类型 | 单个本地音频文件 |
| 支持格式 | `.wav`, `.mp3`, `.m4a`, `.flac`, `.ogg` |
| 采样率 | 任意（内部统一转换为 16kHz） |
| 声道 | 单声道/立体声（内部统一转换为单声道） |
| 时长限制 | 无硬性限制，通过分块策略支持长音频 |

### 输出规范

结构化 JSON 数据，包含每个语音片段的：
- 开始时间 (`start`)
- 结束时间 (`end`)
- 转录文本 (`text`)
- 说话人 ID (`speaker`)

### 核心技术栈

- Python 3.10+
- MLX (Apple Silicon 机器学习框架)
- lightning-whisper-mlx (ASR)
- Senko (Speaker Diarization)
- Silero VAD (语音活动检测)

---

## 3. 架构设计

为最大化 Apple Silicon 芯片的性能，本方案采用模块化的五阶段流水线设计，所有数据尽可能在内存 (RAM) 中流动，以避免磁盘 I/O 成为性能瓶颈。

### 3.1 总体数据流图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              输入处理                                        │
│  [音频文件] → [格式检测] → [重采样 16kHz] → [单声道转换] → [归一化]            │
│                                    ↓                                        │
│                            [float32 PCM 数组]                               │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                    ┌────────────────┴────────────────┐
                    ↓                                 ↓
┌───────────────────────────────────┐  ┌───────────────────────────────────┐
│         ASR 分支                   │  │         SD 分支                    │
│  [Silero VAD] → [语音片段]         │  │  [Senko] → [说话人时间段]          │
│       ↓                           │  │       ↓                           │
│  [lightning-whisper-mlx]          │  │  List[(speaker_id, start, end)]  │
│       ↓                           │  │                                   │
│  [词级时间戳 + 偏移量 → 全局时间戳] │  │                                   │
└───────────────────────────────────┘  └───────────────────────────────────┘
                    │                                 │
                    └────────────────┬────────────────┘
                                     ↓
                    ┌───────────────────────────────────┐
                    │            对齐合并                │
                    │  [IoU 时间轴匹配] → [结构化输出]    │
                    └───────────────────────────────────┘
```

### 3.2 阶段 0: 音频预处理

在所有处理之前，统一音频格式以确保后续模块的兼容性。

| 参数 | 规范值 | 说明 |
|------|--------|------|
| 采样率 | 16000 Hz | Whisper 和 Senko 统一要求 |
| 声道 | Mono | 多声道取平均或指定声道 |
| 格式 | float32 PCM | 内存中统一表示，范围 [-1.0, 1.0] |
| 工具 | `soundfile` + `librosa` | 避免 ffmpeg 子进程开销 |

**处理流程：**

```
输入音频 → 格式检测 → 重采样(16kHz) → 单声道转换 → 归一化 → float32 PCM 数组
```

### 3.3 阶段 1: VAD 检测

| 项目 | 说明 |
|------|------|
| 技术选型 | Silero VAD (ONNX) |
| 作用 | 仅用于 ASR 分支，快速跳过静音段，减少不必要的转录计算 |
| 输出 | 语音片段列表，每个片段包含全局时间戳和对应的音频数据 |

**选型理由：** 极度轻量，毫秒级延迟。过滤长段静音，能有效提升后续转录模型的鲁棒性。

### 3.4 阶段 2: ASR 转录

| 项目 | 说明 |
|------|------|
| 技术选型 | lightning-whisper-mlx |
| 输入 | VAD 输出的语音片段 |
| 输出 | 词级时间戳（经偏移量转换为全局时间戳） |

**选型理由：** 利用 MLX 框架深度优化对 Apple Silicon GPU 的调用，通过批量推理 (Batch Inference) 实现超实时转录。

### 3.5 阶段 3: 声纹分割 (Speaker Diarization)

| 项目 | 说明 |
|------|------|
| 技术选型 | Senko |
| 输入 | **原始完整音频**（非 VAD 切片），让 Senko 使用自己的内置 VAD |
| 输出 | `List[Tuple[speaker_id, start, end]]` |

**关键决策：**

| 决策项 | 选择 | 原因 |
|--------|------|------|
| 输入形式 | 原始完整音频 | Senko 的 CAM++ 嵌入依赖连续上下文，切片可能破坏声纹一致性 |
| VAD 使用 | Senko 内置 VAD | 声纹分割需要更精细的语音边界检测 |
| 与 ASR 的关系 | 并行执行 | 两者无数据依赖，可同时处理提升效率 |

**选型理由：** Senko 基于 3D-Speaker 项目深度优化，在 Mac 上通过 CoreML 运行 CAM++ 模型，实现秒级声纹嵌入提取与聚类。

### 3.6 阶段 4: 时间对齐

| 项目 | 说明 |
|------|------|
| 算法 | 时间轴交集 (IoU) |
| 输入 | ASR 词级时间戳 + SD 说话人时间段 |
| 输出 | 带说话人标签的转录结果 |

**对齐策略：** 将转录模型输出的词级时间戳与分割模型输出的说话人时间段进行数学匹配，实现精准对齐。

---

## 4. 核心机制

### 4.1 时间戳同步机制

解决 VAD 切片后时间戳归零的问题，确保所有时间戳都是相对于原始音频的全局时间。

**核心数据结构：**

```python
from dataclasses import dataclass
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
```

**同步流程：**

```
原始音频 → VAD 检测 → 片段列表 [(start, end), ...]
                           ↓
                    ASR 处理每个片段
                           ↓
                    词时间戳 (局部) + 片段偏移量
                           ↓
                    全局时间戳
```

### 4.2 长音频分块策略

针对超过 30 分钟的长音频，采用滑动窗口分块处理，避免内存溢出。

| 参数 | 值 | 说明 |
|------|-----|------|
| **分块长度** | 10 分钟 | 平衡内存占用与上下文完整性 |
| **重叠区域** | 30 秒 | 避免边界处说话人/词被截断 |
| **去重策略** | 重叠区优先取后块结果 | 后块有更完整的前向上下文 |

**处理流程示例（30分钟音频）：**

```
块 1: [0:00 - 10:30] → 处理 → 保留 [0:00 - 10:00]
块 2: [10:00 - 20:30] → 处理 → 保留 [10:00 - 20:00]
块 3: [20:00 - 30:00] → 处理 → 保留 [20:00 - 30:00]
```

**内存估算（1小时音频）：**

| 组件 | 内存占用 |
|------|----------|
| 原始 16kHz float32 音频 | ~230 MB |
| 单个 10 分钟块 | ~38 MB |
| Whisper large-v3 推理峰值 | ~4 GB |
| **总计安全阈值** | **8 GB 统一内存可处理** |

### 4.3 IoU 对齐算法

**匹配粒度：** 按词级别匹配

**算法逻辑：**

```python
def assign_speaker(word_start: float, word_end: float, 
                   speaker_segments: List[Tuple[str, float, float]]) -> str:
    """
    为单个词分配说话人 ID
    
    Args:
        word_start: 词开始时间
        word_end: 词结束时间
        speaker_segments: 说话人时间段列表 [(speaker_id, start, end), ...]
    
    Returns:
        说话人 ID，若无匹配则返回 "UNKNOWN"
    """
    best_speaker = "UNKNOWN"
    best_iou = 0.0
    
    for speaker_id, seg_start, seg_end in speaker_segments:
        # 计算交集
        intersection_start = max(word_start, seg_start)
        intersection_end = min(word_end, seg_end)
        intersection = max(0, intersection_end - intersection_start)
        
        # 计算并集
        union = (word_end - word_start) + (seg_end - seg_start) - intersection
        
        # 计算 IoU
        iou = intersection / union if union > 0 else 0
        
        if iou > best_iou:
            best_iou = iou
            best_speaker = speaker_id
    
    return best_speaker
```

**冲突处理：** 当词跨越说话人边界时，选择 IoU 最高的说话人。

---

## 5. 技术选型汇总

| 阶段 | 核心任务 | 技术选型 | 运行环境 | 选型理由 |
|------|----------|----------|----------|----------|
| 0. 预处理 | 格式统一 | soundfile + librosa | CPU | 纯 Python，无外部依赖 |
| 1. VAD | 语音活动检测 | Silero VAD (ONNX) | CPU | 极度轻量，毫秒级延迟 |
| 2. ASR | 语音转文字 | lightning-whisper-mlx | GPU/ANE (MLX) | Apple Silicon 原生优化，超实时转录 |
| 3. SD | 声纹分割 | Senko | CoreML | Mac 原生支持，秒级处理 |
| 4. 对齐 | 结果合并 | IoU 匹配 | CPU | 简单高效的数学匹配 |

---

## 6. 输出格式定义

### JSON 输出结构

```json
{
  "metadata": {
    "audio_file": "example.mp3",
    "duration": 3600.5,
    "sample_rate": 16000,
    "num_speakers": 3,
    "processing_time": 45.2
  },
  "speakers": [
    {"id": "SPEAKER_00", "total_duration": 1200.3},
    {"id": "SPEAKER_01", "total_duration": 1800.7},
    {"id": "SPEAKER_02", "total_duration": 599.5}
  ],
  "segments": [
    {
      "start": 0.0,
      "end": 2.5,
      "speaker": "SPEAKER_00",
      "text": "大家好，欢迎来到今天的会议。"
    },
    {
      "start": 2.8,
      "end": 5.2,
      "speaker": "SPEAKER_01",
      "text": "谢谢主持人，我先来汇报一下..."
    }
  ]
}
```

### 字段说明

| 字段 | 类型 | 说明 |
|------|------|------|
| `metadata.audio_file` | string | 输入音频文件名 |
| `metadata.duration` | float | 音频总时长（秒） |
| `metadata.num_speakers` | int | 检测到的说话人数量 |
| `metadata.processing_time` | float | 处理耗时（秒） |
| `segments[].start` | float | 片段开始时间（秒） |
| `segments[].end` | float | 片段结束时间（秒） |
| `segments[].speaker` | string | 说话人 ID |
| `segments[].text` | string | 转录文本 |
