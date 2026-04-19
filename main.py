"""CLI entry point for ASR pipeline."""

import argparse
import os
import sys

from asr.pipeline import run_pipeline
from asr.config import get_api_host, get_api_port


def main():
    parser = argparse.ArgumentParser(
        description="ASR Pipeline - Multi-stage subtitle generation"
    )
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Transcribe command
    trans = subparsers.add_parser("transcribe", help="Run full ASR pipeline")
    trans.add_argument("audio", help="Path to audio file")
    trans.add_argument("-o", "--output", help="Output directory", default=None)
    trans.add_argument("--fmt", choices=["srt", "ass", "all"], default="srt",
                       help="Output format (default: srt)")
    trans.add_argument("--style", default="default",
                       help="ASS style name (default: default)")
    trans.add_argument("--fix-dir", default=None,
                       help="Directory containing fix CSV files")
    trans.add_argument("--language", default=None,
                       help="Language hint (e.g., Chinese, English)")
    trans.add_argument("--model-size", default="1.7B",
                       choices=["1.7B", "0.6B"],
                       help="Model size (default: 1.7B)")
    trans.add_argument("--max-chars", type=int, default=14,
                       help="Max characters per subtitle line (default: 14)")
    trans.add_argument("--resume-from", choices=["asr", "break", "fix", "render"],
                       default=None, help="Resume from a specific stage")

    # Serve command
    serve = subparsers.add_parser("serve", help="Start API server")
    serve.add_argument("--host", default=None, help="Host to bind (default: from config)")
    serve.add_argument("--port", type=int, default=None, help="Port to bind (default: from config)")
    serve.add_argument("--reload", action="store_true", help="Enable auto-reload")

    args = parser.parse_args()

    if args.command == "transcribe":
        audio_path = os.path.abspath(args.audio)
        if not os.path.exists(audio_path):
            print(f"Error: Audio file not found: {audio_path}")
            sys.exit(1)

        output_dir = os.path.abspath(args.output) if args.output else None
        if output_dir is None:
            output_dir = os.path.dirname(audio_path)

        result = run_pipeline(
            audio_path=audio_path,
            output_dir=output_dir,
            fmt=args.fmt,
            ass_style=args.style,
            fix_dir=args.fix_dir,
            language=args.language,
            model_size=args.model_size,
            max_chars=args.max_chars,
            resume_from=args.resume_from,
        )

        if isinstance(result, dict) and "check_errors" in result:
            sys.exit(1)

    elif args.command == "serve":
        import uvicorn
        from api import app

        uvicorn.run(
            "api:app",
            host=args.host if args.host is not None else get_api_host(),
            port=args.port if args.port is not None else get_api_port(),
            reload=args.reload,
        )

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
