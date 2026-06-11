#!/usr/bin/env python3
"""
Fast video compressor - targets a specific output file size.

Usage:
    python compress_video.py input.mov [output.mp4] [--size-mb 90]

Strategy:
- Calculates exact bitrate needed to hit target size based on duration
- Uses hardware acceleration when available (VideoToolbox/NVENC/QSV)
- Falls back to libx264 with `ultrafast` preset for speed
- Optionally downscales to 1080p max (big speed win for screen recordings)

Requires: ffmpeg + ffprobe in PATH.
"""

import argparse
import json
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path


def run(cmd, capture=True):
    """Run a subprocess command and return stdout (or stream output if capture=False)."""
    if capture:
        result = subprocess.run(cmd, capture_output=True, text=True)
        return result
    else:
        return subprocess.run(cmd)


def probe(path: Path) -> dict:
    """Get duration, size, and video stream info via ffprobe."""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration,size",
        "-show_entries", "stream=codec_type,width,height,codec_name",
        "-of", "json",
        str(path),
    ]
    result = run(cmd)
    if result.returncode != 0:
        sys.exit(f"ffprobe failed:\n{result.stderr}")
    data = json.loads(result.stdout)
    duration = float(data["format"]["duration"])
    size_bytes = int(data["format"]["size"])
    video_stream = next(
        (s for s in data["streams"] if s.get("codec_type") == "video"), None
    )
    if not video_stream:
        sys.exit("No video stream found in input.")
    return {
        "duration": duration,
        "size_bytes": size_bytes,
        "width": int(video_stream.get("width", 0)),
        "height": int(video_stream.get("height", 0)),
        "codec": video_stream.get("codec_name", ""),
    }


def detect_hw_encoder() -> tuple[str | None, list[str]]:
    """
    Detect a hardware H.264 encoder available on this system.
    Returns (encoder_name, extra_args) or (None, []) if not available.
    """
    # Check what ffmpeg supports
    result = run(["ffmpeg", "-hide_banner", "-encoders"])
    encoders = result.stdout if result.returncode == 0 else ""

    system = platform.system()

    # macOS - VideoToolbox (very fast, good quality)
    if system == "Darwin" and "h264_videotoolbox" in encoders:
        return "h264_videotoolbox", ["-allow_sw", "1"]

    # NVIDIA - NVENC (fastest on NVIDIA GPUs)
    if "h264_nvenc" in encoders:
        # Quick test if it actually works (driver may be missing)
        test = run([
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-f", "lavfi", "-i", "nullsrc=s=64x64:d=0.1",
            "-c:v", "h264_nvenc", "-f", "null", "-",
        ])
        if test.returncode == 0:
            return "h264_nvenc", ["-preset", "p1", "-tune", "ll"]

    # Intel - QSV
    if "h264_qsv" in encoders:
        test = run([
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-f", "lavfi", "-i", "nullsrc=s=64x64:d=0.1",
            "-c:v", "h264_qsv", "-f", "null", "-",
        ])
        if test.returncode == 0:
            return "h264_qsv", ["-preset", "veryfast"]

    # AMD - AMF (Windows mainly)
    if "h264_amf" in encoders:
        test = run([
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-f", "lavfi", "-i", "nullsrc=s=64x64:d=0.1",
            "-c:v", "h264_amf", "-f", "null", "-",
        ])
        if test.returncode == 0:
            return "h264_amf", ["-quality", "speed"]

    return None, []


def compute_bitrates(target_size_mb: float, duration_s: float, has_audio: bool):
    """
    Given a target file size and duration, compute video and audio bitrates.
    Reserves ~96 kbps for audio if present, and a small overhead for the container.
    Returns (video_bitrate_kbps, audio_bitrate_kbps).
    """
    target_bits = target_size_mb * 1024 * 1024 * 8
    # Container overhead ~1.5%
    target_bits *= 0.985
    audio_kbps = 96 if has_audio else 0
    audio_bits = audio_kbps * 1000 * duration_s
    video_bits = target_bits - audio_bits
    if video_bits <= 0:
        sys.exit("Target size too small for given duration.")
    video_kbps = int(video_bits / 1000 / duration_s)
    return video_kbps, audio_kbps


def has_audio_stream(path: Path) -> bool:
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "a",
        "-show_entries", "stream=codec_type",
        "-of", "csv=p=0",
        str(path),
    ]
    result = run(cmd)
    return "audio" in result.stdout


def build_scale_filter(width: int, height: int, max_height: int = 1080) -> str | None:
    """If video is taller than max_height, scale down preserving aspect ratio."""
    if height <= max_height:
        return None
    # Keep width even (required by yuv420p)
    return f"scale=-2:{max_height}"


def main():
    parser = argparse.ArgumentParser(description="Fast video compressor with size target.")
    parser.add_argument("input", type=Path, help="Input video file")
    parser.add_argument("output", type=Path, nargs="?", help="Output file (default: <input>_compressed.mp4)")
    parser.add_argument("--size-mb", type=float, default=90.0, help="Target size in MB (default: 90)")
    parser.add_argument("--max-height", type=int, default=1080, help="Max output height in pixels (default: 1080)")
    parser.add_argument("--no-hw", action="store_true", help="Disable hardware acceleration")
    parser.add_argument("--no-downscale", action="store_true", help="Don't downscale even if larger than max-height")
    args = parser.parse_args()

    if not args.input.exists():
        sys.exit(f"Input not found: {args.input}")

    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        sys.exit("ffmpeg and ffprobe must be installed and in PATH.")

    output = args.output or args.input.with_name(f"{args.input.stem}_compressed.mp4")

    print(f"📹 Probing input: {args.input.name}")
    info = probe(args.input)
    audio = has_audio_stream(args.input)
    input_mb = info["size_bytes"] / 1024 / 1024
    print(f"   Duration: {info['duration']:.1f}s | Size: {input_mb:.1f} MB | "
          f"Resolution: {info['width']}x{info['height']} | Codec: {info['codec']} | "
          f"Audio: {'yes' if audio else 'no'}")

    # Compute bitrates to hit target size
    v_kbps, a_kbps = compute_bitrates(args.size_mb, info["duration"], audio)
    print(f"🎯 Target: {args.size_mb} MB → video {v_kbps} kbps, audio {a_kbps} kbps")

    # Pick encoder
    if args.no_hw:
        encoder, extra_enc_args = None, []
    else:
        encoder, extra_enc_args = detect_hw_encoder()

    if encoder:
        print(f"⚡ Using hardware encoder: {encoder}")
    else:
        encoder = "libx264"
        extra_enc_args = ["-preset", "ultrafast", "-tune", "fastdecode"]
        print("🐢 Using libx264 (ultrafast)")

    # Build ffmpeg command
    cmd = ["ffmpeg", "-y", "-i", str(args.input)]

    # Video filter (downscale if needed)
    scale = None
    if not args.no_downscale:
        scale = build_scale_filter(info["width"], info["height"], args.max_height)
    if scale:
        print(f"📐 Downscaling: {scale}")
        cmd += ["-vf", scale]

    # Video encoding
    cmd += [
        "-c:v", encoder,
        "-b:v", f"{v_kbps}k",
        "-maxrate", f"{int(v_kbps * 1.5)}k",
        "-bufsize", f"{int(v_kbps * 2)}k",
        *extra_enc_args,
        "-pix_fmt", "yuv420p",
    ]

    # Audio encoding
    if audio:
        cmd += ["-c:a", "aac", "-b:a", f"{a_kbps}k"]
    else:
        cmd += ["-an"]

    # Streaming-friendly mp4
    cmd += ["-movflags", "+faststart", str(output)]

    print(f"🔧 Running: {' '.join(cmd)}\n")

    start = time.time()
    proc = run(cmd, capture=False)
    elapsed = time.time() - start

    if proc.returncode != 0:
        sys.exit(f"\n❌ ffmpeg failed (exit {proc.returncode})")

    out_size_mb = output.stat().st_size / 1024 / 1024
    ratio = input_mb / out_size_mb if out_size_mb else 0
    speed = info["duration"] / elapsed if elapsed else 0
    print(f"\n✅ Done in {elapsed:.1f}s ({speed:.1f}x realtime)")
    print(f"   Output: {output}")
    print(f"   Size: {input_mb:.1f} MB → {out_size_mb:.1f} MB ({ratio:.1f}× smaller)")


if __name__ == "__main__":
    main()