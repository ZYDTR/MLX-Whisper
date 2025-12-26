"""命令行入口

提供命令行接口用于处理音频文件
"""

import argparse
import sys
from pathlib import Path


def main():
    """命令行入口点"""
    parser = argparse.ArgumentParser(
        description="Apple Silicon 本地语音处理流水线",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 基本使用
  python -m src.cli audio.mp3 -o output.json

  # 指定语言和模型
  python -m src.cli audio.mp3 -o output.json --language zh --model large-v3

  # 指定说话人数量
  python -m src.cli meeting.wav -o meeting.json --num-speakers 3

  # 使用量化模型 (减少内存占用)
  python -m src.cli audio.mp3 -o output.json --quantization 4bit
        """
    )
    
    # 必需参数
    parser.add_argument(
        "audio_file",
        type=str,
        help="输入音频文件路径 (支持 .wav, .mp3, .m4a, .flac, .ogg)"
    )
    
    # 输出参数
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="输出 JSON 文件路径 (默认: <输入文件名>.json)"
    )
    
    # ASR 参数
    parser.add_argument(
        "--model",
        type=str,
        default="large-v3",
        choices=[
            "tiny", "tiny.en", "base", "base.en",
            "small", "small.en", "medium", "medium.en",
            "large", "large-v2", "large-v3",
            "distil-large-v2", "distil-large-v3"
        ],
        help="Whisper 模型 (默认: large-v3)"
    )
    
    parser.add_argument(
        "--language",
        type=str,
        default=None,
        help="语言代码，如 zh, en, ja (默认: 自动检测)"
    )
    
    parser.add_argument(
        "--quantization",
        type=str,
        default=None,
        choices=["4bit", "8bit"],
        help="模型量化选项 (默认: 不量化)"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=12,
        help="批处理大小 (默认: 12)"
    )
    
    # 声纹分割参数
    parser.add_argument(
        "--num-speakers",
        type=int,
        default=None,
        help="说话人数量 (默认: 自动检测)"
    )
    
    parser.add_argument(
        "--min-speakers",
        type=int,
        default=1,
        help="最小说话人数量 (默认: 1)"
    )
    
    parser.add_argument(
        "--max-speakers",
        type=int,
        default=10,
        help="最大说话人数量 (默认: 10)"
    )
    
    # VAD 参数
    parser.add_argument(
        "--vad-threshold",
        type=float,
        default=0.5,
        help="VAD 阈值 0-1 (默认: 0.5)"
    )
    
    # 处理参数
    parser.add_argument(
        "--no-parallel",
        action="store_true",
        help="禁用并行处理"
    )
    
    # 调试参数
    parser.add_argument(
        "--info-only",
        action="store_true",
        help="仅显示音频信息，不进行处理"
    )
    
    args = parser.parse_args()
    
    # 检查输入文件
    audio_path = Path(args.audio_file)
    if not audio_path.exists():
        print(f"❌ 错误: 文件不存在: {audio_path}")
        sys.exit(1)
    
    # 仅显示信息模式
    if args.info_only:
        from .preprocessor import get_audio_info
        info = get_audio_info(str(audio_path))
        print("\n📊 音频信息:")
        for key, value in info.items():
            print(f"   {key}: {value}")
        return
    
    # 设置输出路径
    output_path = args.output
    if output_path is None:
        output_path = str(audio_path.with_suffix('.json'))
    
    # 导入并运行流水线
    from .pipeline import SpeechPipeline, PipelineConfig
    
    config = PipelineConfig(
        whisper_model=args.model,
        language=args.language,
        batch_size=args.batch_size,
        quantization=args.quantization,
        num_speakers=args.num_speakers,
        min_speakers=args.min_speakers,
        max_speakers=args.max_speakers,
        vad_threshold=args.vad_threshold,
        parallel_processing=not args.no_parallel
    )
    
    print("\n" + "=" * 60)
    print("🎙️  Apple Silicon 本地语音处理流水线")
    print("=" * 60 + "\n")
    
    pipeline = SpeechPipeline(config)
    result = pipeline.process(str(audio_path), output_path)
    
    # 打印摘要
    print("\n" + "=" * 60)
    print("📝 转录摘要 (前 5 段)")
    print("=" * 60)
    
    for i, seg in enumerate(result.segments[:5]):
        speaker = seg.speaker
        text = seg.text[:50] + "..." if len(seg.text) > 50 else seg.text
        print(f"\n[{seg.start:.1f}s - {seg.end:.1f}s] {speaker}")
        print(f"  {text}")
    
    if len(result.segments) > 5:
        print(f"\n... 还有 {len(result.segments) - 5} 个片段")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()

