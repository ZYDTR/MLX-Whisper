# Pyannote 声纹分割设置指南

## 问题说明

pyannote.audio 模型 `pyannote/speaker-diarization-3.1` 是 HuggingFace 上的受限制模型（gated repository），需要：
1. 接受用户条件
2. 提供 HuggingFace token 进行认证

## 解决步骤

### 1. 接受用户条件

访问以下链接并接受用户条件：
```
https://hf.co/pyannote/speaker-diarization-3.1
```

### 2. 创建 HuggingFace Token

1. 访问 https://hf.co/settings/tokens
2. 点击 "New token"
3. 选择 "Read" 权限
4. 复制生成的 token（格式：`hf_xxxxxxxxxxxxx`）

### 3. 配置 Token

有两种方式配置 token：

#### 方式 1: 环境变量（推荐）

```bash
export HF_TOKEN=your_token_here
```

或在 `.bashrc` / `.zshrc` 中添加：
```bash
export HF_TOKEN=your_token_here
```

#### 方式 2: 代码中传入

```python
from src.pipeline import PipelineConfig, SpeechPipeline

config = PipelineConfig(
    # ... 其他配置
)

# 在创建 diarizer 时传入 token
pipeline = SpeechPipeline(config)
pipeline.diarizer.hf_token = "your_token_here"
```

或在初始化时：

```python
from src.diarization import SpeakerDiarizer

diarizer = SpeakerDiarizer(
    num_speakers=None,
    hf_token="your_token_here"
)
```

## 验证安装

运行以下命令验证：

```bash
python3 -c "
from src.diarization import SpeakerDiarizer
import numpy as np

diarizer = SpeakerDiarizer()
test_audio = np.random.randn(16000).astype(np.float32) * 0.1
segments = diarizer.diarize(test_audio)
print('✅ pyannote 加载成功！')
"
```

如果看到 "📥 加载 pyannote 声纹分割模型..." 且没有错误，说明配置成功。

## 注意事项

- Token 需要 "Read" 权限即可
- Token 不要提交到代码仓库
- 如果未配置 token，系统会自动回退到简化的声纹分割（基于能量检测）



