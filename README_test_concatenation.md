# 拼接方案测试脚本使用说明

## 📋 脚本说明

`test_concatenation_solution.py` 实现了拼接方案测试，使用以下配置：
- **方案**: 拼接长音频（Concatenation）
- **模型**: Large-v3-turbo (lightning-whisper-mlx)
- **batch_size**: 4（最佳稳定性与速度平衡）
- **输出格式**: 纯文本（speaker + 说话内容）

## 🚀 使用方法

### 基本用法

```bash
python3 test_concatenation_solution.py <音频文件路径>
```

### 指定输出文件

```bash
python3 test_concatenation_solution.py <音频文件路径> -o <输出文件.txt>
```

### 示例

```bash
# 使用默认输出文件名（音频文件名.txt）
python3 test_concatenation_solution.py "20251205 234222-BF444D4E_part_000.m4a"

# 指定输出文件名
python3 test_concatenation_solution.py "audio.mp3" -o "output.txt"
```

## 📝 输出格式

输出为纯文本格式，每行一个说话片段：

```
SPEAKER_00: 第一段说话内容
SPEAKER_01: 第二段说话内容
SPEAKER_00: 第三段说话内容
...
```

## ⚙️ 配置参数

脚本中的关键参数（可在代码中修改）：

```python
MODEL_NAME = "large-v3"      # 模型名称
BATCH_SIZE = 4               # Batch 大小（Large-v3 推荐 4）
CONTEXT_DURATION = 3.0       # 上下文时长（秒）
MERGE_GAP = 2.0              # 片段合并间隔（秒）
```

## 🔍 处理流程

1. **加载音频** - 读取并预处理音频文件
2. **VAD 检测** - 检测语音活动片段
3. **拼接片段** - 将所有片段拼接成长音频
4. **并行处理** - 同时进行 ASR 转录和声纹分割
5. **时间对齐** - 对齐转录结果和说话人信息
6. **输出文本** - 生成纯文本格式输出

## 📊 性能指标

脚本会输出以下性能指标：
- 总处理时间
- 实时比（音频时长/处理时间）
- 识别说话人数
- 转录片段数

## ⚠️ 注意事项

1. **内存要求**: Large-v3 模型需要较大内存，建议至少 16GB
2. **首次运行**: 首次运行会下载模型，需要较长时间
3. **音频格式**: 支持常见音频格式（mp3, m4a, wav 等）

## 🐛 故障排除

### 模型加载失败
- 检查网络连接（需要下载模型）
- 检查磁盘空间（模型约 3-6GB）

### 内存不足
- 尝试减小 batch_size（改为 2 或 1）
- 使用 smaller 模型（如 "small"）

### 输出为空
- 检查音频是否包含语音
- 调整 VAD 阈值参数



