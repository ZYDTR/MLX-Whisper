#!/bin/bash
# MLX Whisper 语音转录 UI 启动脚本
# 双击此文件即可启动 UI

# 切换到脚本所在目录
cd "$(dirname "$0")"

# 加载环境变量（包括 HF_TOKEN）
if [ -f ~/.zshrc ]; then
    source ~/.zshrc 2>/dev/null || true
fi

echo "========================================"
echo "🎤 MLX Whisper 语音转录 UI"
echo "========================================"
echo ""

# ===== 1. 查找 Python 3.10+ =====
SYSTEM_PYTHON=""

# 按优先级查找：python3.12 > python3.13 > python3.11 > python3.10
# (python3.12 通常最稳定)
for py_version in python3.12 python3.13 python3.11 python3.10; do
    if command -v $py_version &> /dev/null; then
        SYSTEM_PYTHON=$(command -v $py_version)
        break
    fi
done

# 检查 Homebrew 安装的路径
if [ -z "$SYSTEM_PYTHON" ]; then
    for py_path in /opt/homebrew/bin/python3.12 /opt/homebrew/bin/python3.13 /opt/homebrew/bin/python3.11 /opt/homebrew/bin/python3.10 \
                   /usr/local/bin/python3.12 /usr/local/bin/python3.13 /usr/local/bin/python3.11 /usr/local/bin/python3.10; do
        if [ -f "$py_path" ]; then
            SYSTEM_PYTHON="$py_path"
            break
        fi
    done
fi

# 如果没找到 Python 3.10+，提示错误
if [ -z "$SYSTEM_PYTHON" ]; then
    echo "❌ 错误: 未找到 Python 3.10+"
    echo "   当前系统 Python: $(python3 --version 2>/dev/null || echo '未安装')"
    echo ""
    echo "   请安装 Python 3.10 或更高版本："
    echo "   brew install python@3.12"
    read -p "按任意键退出..."
    exit 1
fi

echo "✅ 系统 Python: $($SYSTEM_PYTHON --version)"

# ===== 2. 设置虚拟环境 =====
VENV_DIR=".venv"

# 检查虚拟环境是否存在且有效
if [ -f "$VENV_DIR/bin/python" ]; then
    echo "✅ 虚拟环境已存在: $VENV_DIR"
    VENV_PYTHON_VERSION=$($VENV_DIR/bin/python --version 2>/dev/null || echo "未知")
    echo "   版本: $VENV_PYTHON_VERSION"
else
    echo ""
    echo "📦 首次运行，正在创建虚拟环境..."
    echo "   使用: $SYSTEM_PYTHON"
    $SYSTEM_PYTHON -m venv "$VENV_DIR"
    if [ $? -ne 0 ]; then
        echo "❌ 创建虚拟环境失败"
        read -p "按任意键退出..."
        exit 1
    fi
    echo "✅ 虚拟环境创建成功"
fi

# 激活虚拟环境
source "$VENV_DIR/bin/activate"
PYTHON_CMD="$VENV_DIR/bin/python"

echo "✅ 使用 Python: $($PYTHON_CMD --version)"
echo "   路径: $PYTHON_CMD"

# ===== 3. 检查并安装依赖 =====
echo ""
echo "📦 检查依赖..."

# 升级 pip（静默）
$PYTHON_CMD -m pip install --upgrade pip -q 2>/dev/null

# 检查核心依赖
MISSING_DEPS=()

if ! $PYTHON_CMD -c "import gradio" 2>/dev/null; then
    MISSING_DEPS+=("gradio")
fi

if ! $PYTHON_CMD -c "import mlx_whisper" 2>/dev/null; then
    MISSING_DEPS+=("mlx-whisper")
fi

if ! $PYTHON_CMD -c "import librosa" 2>/dev/null; then
    MISSING_DEPS+=("librosa")
fi

# 检查 VAD 依赖 (onnxruntime for Silero VAD)
VAD_OK=false
if $PYTHON_CMD -c "import onnxruntime" 2>/dev/null; then
    VAD_OK=true
else
    MISSING_DEPS+=("onnxruntime")
fi

# 检查说话人分离依赖 (pyannote)
PYANNOTE_OK=false
if $PYTHON_CMD -c "from pyannote.audio import Pipeline" 2>/dev/null; then
    PYANNOTE_OK=true
fi

# 安装缺失的依赖
if [ ${#MISSING_DEPS[@]} -gt 0 ]; then
    echo "   ⚠️  缺少依赖: ${MISSING_DEPS[*]}"
    echo ""
    echo "📥 正在安装依赖（首次可能需要几分钟）..."
    
    # 安装缺失的依赖
    $PYTHON_CMD -m pip install "${MISSING_DEPS[@]}" --quiet
    
    if [ $? -ne 0 ]; then
        echo "❌ 依赖安装失败"
        echo "   请尝试手动安装："
        echo "   source $VENV_DIR/bin/activate"
        echo "   pip install ${MISSING_DEPS[*]}"
        read -p "按任意键退出..."
        exit 1
    fi
    
    echo "✅ 依赖安装完成"
else
    echo "✅ 核心依赖已就绪"
    echo "   gradio: $($PYTHON_CMD -c "import gradio; print(gradio.__version__)" 2>/dev/null || echo 'N/A')"
fi

# 显示功能状态 (安装后重新检查)
echo ""
echo "📋 功能状态:"

# VAD 状态 (重新检查)
if $PYTHON_CMD -c "import onnxruntime" 2>/dev/null; then
    echo "   ✅ VAD (语音活动检测): Silero VAD 已就绪"
else
    echo "   ⚠️  VAD: onnxruntime 未安装，VAD 功能将不可用"
fi

# 说话人分离状态
if [ "$PYANNOTE_OK" = true ]; then
    echo "   ✅ 说话人分离: pyannote.audio 已安装"
else
    echo "   ⚠️  说话人分离: pyannote.audio 未安装 (将使用简化模式)"
    echo "      如需完整功能: pip install pyannote.audio torch torchaudio"
fi

# ===== 4. 检查 HF_TOKEN =====
echo ""
if [ -n "$HF_TOKEN" ]; then
    echo "✅ 已检测到 HF_TOKEN (说话人分离可用)"
else
    echo "⚠️  未设置 HF_TOKEN (说话人分离将使用简化模式)"
    echo "   如需完整功能，请设置: export HF_TOKEN=your_token"
fi

# ===== 5. 检查端口 =====
echo ""
echo "🔍 检查端口 7860..."

PORT_PID=$(lsof -ti:7860 2>/dev/null)
if [ -n "$PORT_PID" ]; then
    echo "⚠️  端口 7860 已被占用 (PID: $PORT_PID)，正在清理..."
    kill -9 $PORT_PID 2>/dev/null
    sleep 2
    PORT_PID=$(lsof -ti:7860 2>/dev/null)
    if [ -n "$PORT_PID" ]; then
        echo "❌ 无法清理端口，请手动关闭占用进程"
        echo "   运行: lsof -ti:7860 | xargs kill -9"
        read -p "按任意键退出..."
        exit 1
    else
        echo "✅ 端口已清理"
    fi
else
    echo "✅ 端口 7860 可用"
fi

# ===== 6. 启动 UI =====
echo ""
echo "🚀 启动 UI..."
echo "   浏览器将自动打开: http://127.0.0.1:7860"
echo ""
echo "   按 Ctrl+C 停止服务器"
echo "========================================"
echo ""

# 启动 UI
$PYTHON_CMD ui/app.py

# 如果正常退出，等待用户确认
echo ""
echo "服务器已停止"
read -p "按任意键关闭窗口..."
