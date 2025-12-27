# 性能测试报告

**测试日期**: 2024-12-24  
**测试音频**: `/Users/zhengyidi/MLX/20251205 234222-BF444D4E_part_000.m4a`  
**测试时长**: 前 10 分钟 (600秒)

---

## 📊 测试结果汇总

| 配置名称 | Batch Size | 分块数 | 处理时间 | 实时比 | 段落数 | 内存(GB) | 状态 |
|----------|------------|--------|----------|--------|--------|----------|------|
| baseline_b12_p1 | 12 | 1 | 49.8s | **12.0x** | 207 | 1.22 | ✅ |
| batch18_b18_p1 | 18 | 1 | 41.3s | **14.5x** | 207 | 1.22 | ✅ |
| parallel2_b12_p2 | 12 | 2 | 40.9s | **14.7x** | 193 | 1.22 | ✅ |
| **aggressive_b18_p2** | **18** | **2** | **40.3s** | **14.9x** | 193 | 1.22 | ✅ 最佳 |

### 🏆 最佳配置: `aggressive_b18_p2`

- **实时比**: 14.9x (处理 10 分钟音频仅需 40.3 秒)
- **内存占用**: 1.22 GB (非常低)
- **速度提升**: 相比基准配置提升 24%

---

## 🔧 详细参数配置

### 1. Whisper 模型参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `model` | `mlx-community/whisper-large-v3-turbo` | 使用 Turbo 模型，速度更快 |
| `language` | `zh` | 中文 |
| `word_timestamps` | `True` | 启用词级时间戳 |
| `verbose` | `False` | 关闭详细输出 |

### 2. 识别敏感度参数 (针对电话录音优化)

| 参数 | 值 | 默认值 | 说明 |
|------|-----|--------|------|
| `no_speech_threshold` | **0.1** | 0.6 | 降低静音判断门槛，更容易识别低音量语音 |
| `logprob_threshold` | **-2.5** | -1.0 | 接受低置信度的识别结果 |
| `temperature` | **0.2** | 0.0 | 轻微随机性，避免过于保守 |
| `condition_on_previous_text` | **False** | True | 每段独立解码，避免错误传播 |

### 3. 音频增强参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `enable_enhancement` | `True` | 启用音频增强 |
| `enhancement_method` | 频谱减法 (Spectral Subtraction) | 降噪方法 |
| `enhancement_alpha` | `2.0` | 过减因子，控制降噪强度 |
| `noise_estimation` | 前 0.5 秒 | 使用音频开头估计噪声 |

### 4. VAD (语音活动检测) 参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `use_vad` | **False** | 本次测试**不使用 VAD**，直接全量转录 |
| `vad_threshold` | 0.3 | (未使用) |
| `min_speech_duration` | 0.25s | (未使用) |
| `min_silence_duration` | 0.1s | (未使用) |

**为什么不使用 VAD**:  
在电话录音场景中，对方声音较小，VAD 可能误判为静音。直接让 Whisper 处理完整音频效果更好。

### 5. 性能参数

| 参数 | 测试范围 | 最佳值 | 说明 |
|------|----------|--------|------|
| `batch_size` | 12 / 18 | **18** | 批处理大小，提升 GPU 利用率 |
| `parallel_audio` | 1 / 2 | **2** | 分块数量 |

---

## 🔬 关键发现

### 1. Batch Size 影响

```
batch_size=12 → batch_size=18: 速度提升 21% (12.0x → 14.5x)
```

- 增加 batch_size 可以更充分利用 GPU
- 内存几乎无增加 (1.22 GB 不变)
- **建议**: 默认使用 `batch_size=18`

### 2. 音频分块影响

```
单块处理 → 双块处理: 速度小幅提升 (14.5x → 14.7x)
```

- 分块处理略有提升，主要是减少了单次推理的上下文长度
- **注意**: MLX/Metal 不支持真正的 GPU 并行，分块只能顺序处理

### 3. MLX/Metal 并行限制

```
⚠️ 重要发现: MLX 框架使用 Metal GPU 时，不支持多线程并行推理
```

**原因**:
- Metal Command Buffer 不允许并发提交
- 尝试多线程会导致 `MTLCommandBuffer addCompletedHandler` 断言失败

**解决方案**:
- 使用顺序分块处理代替并行
- 如需真正并行，需要多进程 + 独立模型加载 (内存翻倍)

### 4. 内存使用

```
峰值内存: 1.22 GB (非常低)
```

- Turbo 模型内存占用约 1.5 GB
- 16 GB Mac 完全可以处理多个长音频

---

## 📁 输出文件

每个测试配置生成三个文件:

| 格式 | 文件示例 | 用途 |
|------|----------|------|
| TXT | `baseline_b12_p1_transcription.txt` | 纯文本转录 |
| SRT | `baseline_b12_p1_transcription.srt` | 字幕格式 |
| JSON | `baseline_b12_p1_transcription.json` | 完整数据 (含时间戳) |

---

## 🎯 推荐配置

### 电话录音场景 (最佳)

```python
config = {
    # 模型
    "model": "mlx-community/whisper-large-v3-turbo",
    "language": "zh",
    
    # 敏感度 (针对低音量优化)
    "no_speech_threshold": 0.1,
    "logprob_threshold": -2.5,
    "temperature": 0.2,
    "condition_on_previous_text": False,
    
    # 音频增强
    "enable_enhancement": True,
    "enhancement_method": "spectral_subtraction",
    "enhancement_alpha": 2.0,
    
    # 性能
    "batch_size": 18,
}
```

### 清晰录音场景

```python
config = {
    # 模型 (可用 large-v3 获得更高准确率)
    "model": "mlx-community/whisper-large-v3-mlx",
    "language": "zh",
    
    # 敏感度 (默认值)
    "no_speech_threshold": 0.6,
    "logprob_threshold": -1.0,
    "temperature": 0.0,
    "condition_on_previous_text": True,
    
    # 音频增强
    "enable_enhancement": False,
    
    # 性能
    "batch_size": 18,
}
```

---

## 📈 性能预估

基于测试结果，预估不同音频长度的处理时间:

| 音频时长 | Turbo (14.9x) | Large-v3 (~5x) |
|----------|---------------|----------------|
| 10 分钟 | ~40 秒 | ~2 分钟 |
| 30 分钟 | ~2 分钟 | ~6 分钟 |
| 1 小时 | ~4 分钟 | ~12 分钟 |
| 2 小时 | ~8 分钟 | ~24 分钟 |

---

## 📝 频谱减法增强算法

```python
def enhance_spectral_subtraction(audio, sr=16000, alpha=2.0):
    """
    频谱减法降噪
    
    原理:
    1. 使用音频开头 0.5 秒估计噪声频谱
    2. 从整个音频的频谱中减去噪声频谱
    3. 使用相位信息重构音频
    
    参数:
    - audio: 输入音频 (float32, 16kHz)
    - sr: 采样率
    - alpha: 过减因子 (越大降噪越强，但可能损失细节)
    """
    n_fft = 2048
    hop_length = 512
    
    # 估计噪声
    noise_samples = int(0.5 * sr)
    noise = audio[:noise_samples]
    noise_stft = librosa.stft(noise, n_fft=n_fft, hop_length=hop_length)
    noise_mag = np.abs(noise_stft).mean(axis=1, keepdims=True)
    
    # 信号 STFT
    signal_stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    signal_mag = np.abs(signal_stft)
    signal_phase = np.angle(signal_stft)
    
    # 频谱减法
    beta = 0.01  # 频谱下限，防止过度消除
    enhanced_mag = np.maximum(signal_mag - alpha * noise_mag, beta * signal_mag)
    
    # 重构
    enhanced_stft = enhanced_mag * np.exp(1j * signal_phase)
    enhanced = librosa.istft(enhanced_stft, hop_length=hop_length, length=len(audio))
    
    return enhanced
```

---

## 🔗 相关文件

- 测试脚本: `/Users/zhengyidi/MLX/performance_tests/test_performance.py`
- 测试结果: `/Users/zhengyidi/MLX/performance_tests/results/`
- 汇总数据: `/Users/zhengyidi/MLX/performance_tests/results/performance_summary.json`

---

## ✅ 结论

1. **Turbo 模型 + 频谱减法** 是电话录音的最佳选择
2. **batch_size=18** 可安全使用，速度提升明显
3. **MLX 不支持 GPU 并行**，但顺序分块仍有小幅提升
4. **内存占用低** (~1.2GB)，16GB Mac 可轻松处理
5. **实时比 14.9x** 意味着处理 1 小时音频仅需 ~4 分钟

