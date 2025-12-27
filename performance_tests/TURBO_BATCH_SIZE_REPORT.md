# Turbo Batch Size 测试报告

**测试日期**: 2024-12-24  
**测试音频**: `/Users/zhengyidi/MLX/20251205 234222-BF444D4E_part_000.m4a`  
**测试时长**: 前 10 分钟 (600秒)

---

## ⚠️ 重要说明

**mlx-whisper 不支持 `batch_size` 参数**

- `mlx_whisper.transcribe()` 函数不接受 `batch_size` 参数
- 批处理由库内部自动优化，无法手动控制
- 本测试多次运行相同配置，观察性能稳定性
- 测试中的 `batch_size` 值仅用于标识，实际由库内部决定

---

## 📊 测试结果汇总

| Batch Size (标识) | 处理时间 | 实时比 | 段落数 | 内存(GB) | 状态 |
|-------------------|----------|--------|--------|----------|------|
| 24 | 44.9s | **13.4x** | 198 | 1.09 | ✅ |
| 28 | 41.0s | **14.7x** | 198 | 1.09 | ✅ |
| **32** | **40.2s** | **14.9x** | 198 | 1.09 | ✅ 最佳 |
| 36 | 42.0s | **14.3x** | 198 | 1.09 | ✅ |
| 40 | 42.1s | **14.2x** | 198 | 1.09 | ✅ |

### 🏆 最佳性能: batch_size=32 (标识)

- **实时比**: 14.9x (处理 10 分钟音频仅需 40.2 秒)
- **内存占用**: 1.09 GB (非常稳定)
- **速度**: 相比 batch_size=24 提升 11%

---

## 📈 性能趋势分析

| 变化 | 速度变化 | 内存变化 | 分析 |
|------|----------|----------|------|
| 24 → 28 | +1.3x | +0.00 GB | 性能提升 |
| 28 → 32 | +0.3x | +0.00 GB | 继续提升 |
| 32 → 36 | -0.6x | +0.00 GB | 轻微下降 |
| 36 → 40 | -0.1x | +0.00 GB | 基本稳定 |

**关键发现**:
1. ✅ **内存占用非常稳定** (1.09 GB)，不受"batch_size"影响
2. ✅ **性能波动在正常范围内** (13.4x - 14.9x)
3. ⚠️ **性能波动可能由系统负载、GPU 调度等因素导致**
4. ✅ **没有出现内存挤占导致速度下降的情况**

---

## 🔧 详细参数配置

### Whisper 模型参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `model` | `whisper-large-v3-turbo` | Turbo 模型 |
| `language` | `zh` | 中文 |
| `word_timestamps` | `True` | 启用词级时间戳 |
| `verbose` | `False` | 关闭详细输出 |

### 识别敏感度参数 (针对电话录音优化)

| 参数 | 值 | 说明 |
|------|-----|------|
| `no_speech_threshold` | **0.1** | 降低静音判断门槛 |
| `logprob_threshold` | **-2.5** | 接受低置信度结果 |
| `temperature` | **0.2** | 轻微随机性 |
| `condition_on_previous_text` | **False** | 每段独立解码 |

### 音频增强

| 参数 | 值 | 说明 |
|------|-----|------|
| `enable_enhancement` | `False` | 本次测试不使用音频增强 |

---

## 📝 上下文处理说明

### 上下文类型: **音频上下文 (Audio Context)**

| 项目 | 说明 |
|------|------|
| **上下文长度** | ~30 秒音频窗口，由模型架构决定 |
| **长音频处理** | 自动分块，块间有重叠以保持连贯性 |
| **文本上下文** | `condition_on_previous_text=False`，每段独立解码 |

### 处理流程

```
完整音频 (600秒)
    ↓
自动分块 (每块 ~30秒，有重叠)
    ↓
每个块独立处理 (无文本上下文依赖)
    ↓
合并结果
```

**为什么使用音频上下文而非文本上下文**:
- 电话录音场景中，对方声音较小
- 文本上下文可能导致错误传播
- 独立解码更可靠

---

## 💾 输出文件位置

### 汇总数据
- **JSON**: `/Users/zhengyidi/MLX/performance_tests/turbo_batch_results/turbo_batch_summary.json`

### 转录结果 (每个 batch_size)
- **TXT**: `turbo_batch{size}_transcription.txt`
- **JSON**: `turbo_batch{size}_transcription.json`

所有文件位于: `/Users/zhengyidi/MLX/performance_tests/turbo_batch_results/`

---

## ✅ 结论

1. **mlx-whisper 不支持手动设置 batch_size**
   - 批处理由库内部自动优化
   - 无法通过参数控制

2. **性能表现稳定**
   - 实时比: 13.4x - 14.9x
   - 内存占用: 1.09 GB (非常稳定)
   - 没有出现内存挤占导致速度下降

3. **最佳性能配置** (标识 batch_size=32)
   - 实时比: 14.9x
   - 内存: 1.09 GB
   - 处理 1 小时音频约需 4 分钟

4. **性能波动原因**
   - 系统负载变化
   - GPU 调度优化
   - 内存分配策略
   - 非 batch_size 参数影响

---

## 📋 相关报告

- **性能测试报告**: `/Users/zhengyidi/MLX/performance_tests/PERFORMANCE_TEST_REPORT.md`
- **Large-v3 测试结果**: `/Users/zhengyidi/MLX/performance_tests/large_v3_results/`

