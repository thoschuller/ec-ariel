#!/usr/bin/env python3
import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

from watchdog.events import FileSystemEventHandler, FileCreatedEvent, FileMovedEvent
from watchdog.observers import Observer

# --- Config ---
VIDEO_EXTS = {".mp4", ".m4v", ".mov", ".mkv", ".avi"}
STABLE_CHECKS = 3
STABLE_INTERVAL = 1.5
DEFAULT_LEGACY = {"mp4v", "mjpeg"}  # mpeg4 wordt genormaliseerd naar mp4v

# --- Helpers ---
def run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

def ffprobe_codec(path: Path) -> Optional[str]:
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=codec_name",
        "-of", "json",
        str(path),
    ]
    p = run(cmd)
    if p.returncode != 0:
        return None
    try:
        data = json.loads(p.stdout)
        streams = data.get("streams", [])
        if not streams:
            return None
        return streams[0].get("codec_name")
    except Exception:
        return None

def ffprobe_audio_codec(path: Path) -> Optional[str]:
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "a:0",
        "-show_entries", "stream=codec_name",
        "-of", "json",
        str(path),
    ]
    p = run(cmd)
    if p.returncode != 0:
        return None
    try:
        data = json.loads(p.stdout)
        streams = data.get("streams", [])
        if not streams:
            return None
        return streams[0].get("codec_name")
    except Exception:
        return None

def normalize_codec(name: Optional[str]) -> str:
    if not name:
        return "unknown"
    name = name.lower()
    if name == "mpeg4":
        return "mp4v"
    if name in {"mjpeg", "msmpeg4v2", "msmpeg4v3"}:
        return name
    return name

def parse_codec_list(s: Optional[str]) -> set[str]:
    if not s:
        return set()
    items = {normalize_codec(x.strip()) for x in s.split(",") if x.strip()}
    return {x for x in items if x != "unknown"}

def swap_ext_keep_last_dot(p: Path, new_suffix: str, outdir: Optional[Path] = None) -> Path:
    name = p.name
    if name.startswith(".") and name.count(".") == 1:
        base = name
    else:
        base = name.rsplit(".", 1)[0]
    return (outdir or p.parent) / (base + new_suffix)

def is_size_stable(path: Path) -> bool:
    last = -1
    stable = 0
    while True:
        try:
            size = path.stat().st_size
        except FileNotFoundError:
            return False
        if size == last and size > 0:
            stable += 1
            if stable >= STABLE_CHECKS:
                return True
        else:
            stable = 0
        last = size
        time.sleep(STABLE_INTERVAL)

def should_handle(path: Path) -> bool:
    return path.suffix.lower() in VIDEO_EXTS

def has_encoder(name_substring: str) -> bool:
    enc = run(["ffmpeg", "-hide_banner", "-encoders"])
    return enc.returncode == 0 and (name_substring in enc.stdout)

# --- Transcoding ---
def build_ffmpeg_cmd(
    src: Path,
    dst: Path,
    target: str,
    crf: Optional[int],
    preset: Optional[str],
    two_pass: bool,
) -> list[list[str]]:
    if target == "hevc":
        vcodec = "libx265"
        vtag = "hvc1"
        crf = 23 if crf is None else crf
        preset = "medium" if preset is None else preset
        vopts = ["-c:v", vcodec, "-tag:v", vtag, "-crf", str(crf), "-preset", preset]
        passes = 1
    elif target == "av1":
        if has_encoder("libsvtav1"):
            vcodec = "libsvtav1"
            vtag = "av01"
            crf = 28 if crf is None else crf
            preset = "8" if preset is None else preset  # 0=traag..13=snel
            vopts = ["-c:v", vcodec, "-tag:v", vtag, "-crf", str(crf), "-preset", preset]
            passes = 1
        else:
            vcodec = "libaom-av1"
            vtag = "av01"
            crf = 30 if crf is None else crf
            cpu_used = preset if (preset and preset.isdigit()) else "3"  # 0..8
            vopts = ["-c:v", vcodec, "-tag:v", vtag, "-crf", str(crf), "-cpu-used", cpu_used]
            passes = 2 if two_pass else 1
    else:
        raise ValueError("target must be 'hevc' or 'av1'")

    acodec_src = ffprobe_audio_codec(src)
    if acodec_src and acodec_src.lower() in {"aac", "libfdk_aac", "opus"}:
        aopts = ["-c:a", "copy"]
    else:
        aopts = ["-c:a", "aac", "-b:a", "128k"]

    common = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(src),
        "-map", "0",
        "-movflags", "+faststart",
        "-pix_fmt", "yuv420p",
        "-c:s", "copy",
    ]

    if passes == 1:
        return [common + vopts + aopts + [str(dst)]]
    else:
        passlog = dst.with_suffix(dst.suffix + ".log")
        cmd1 = common + ["-an"] + vopts + ["-pass", "1", "-passlogfile", str(passlog), "-f", "mp4", os.devnull]
        cmd2 = common + vopts + aopts + ["-pass", "2", "-passlogfile", str(passlog), str(dst)]
        return [cmd1, cmd2]

def transcode(
    src: Path,
    target: str,
    outdir: Optional[Path],
    crf: Optional[int],
    preset: Optional[str],
    two_pass: bool,
) -> Optional[Path]:
    outdir = outdir or src.parent
    suffix = ".hevc.mp4" if target == "hevc" else ".av1.mp4"
    dst = swap_ext_keep_last_dot(src, suffix, outdir)

    if dst.exists():
        print(f"[skip] Destination exists: {dst}")
        return None

    cmds = build_ffmpeg_cmd(src, dst, target, crf, preset, two_pass)
    for i, cmd in enumerate(cmds, start=1):
        print(f"[ffmpeg] pass {i}/{len(cmds)}: {' '.join(shlex.quote(c) for c in cmd)}")
        p = run(cmd)
        if p.returncode != 0:
            try:
                if dst.exists():
                    dst.unlink()
            except Exception:
                pass
            for ext in (".log", ".log.mbtree", ".log.tmp"):
                maybe = Path(str(dst) + ext)
                if maybe.exists():
                    try:
                        maybe.unlink()
                    except Exception:
                        pass
            print(f"[error] ffmpeg failed:\n{p.stderr}", file=sys.stderr)
            return None

    for ext in (".log", ".log.mbtree", ".log.tmp"):
        maybe = Path(str(dst) + ext)
        if maybe.exists():
            try:
                maybe.unlink()
            except Exception:
                pass

    print(f"[done] {src.name} -> {dst.name}")
    return dst

# --- Watcher ---
def process_if_needed(
    path: Path,
    target: str,
    outdir: Optional[Path],
    crf: Optional[int],
    preset: Optional[str],
    two_pass: bool,
    only_codecs: set[str],
):
    if not path.exists() or not should_handle(path):
        return

    print(f"[watch] Detected {path.name}, waiting for stability...")
    if not is_size_stable(path):
        print(f"[skip] File disappeared or never stabilized: {path}")
        return

    raw = ffprobe_codec(path)
    vcodec = normalize_codec(raw)
    print(f"[probe] {path.name} video codec: {raw} -> {vcodec}")

    target_codecs = only_codecs if only_codecs else DEFAULT_LEGACY
    if vcodec not in target_codecs:
        print(f"[skip] Not in target codecs {sorted(target_codecs)}: {path.name}")
        return

    transcode(path, target=target, outdir=outdir, crf=crf, preset=preset, two_pass=two_pass)

class Handler(FileSystemEventHandler):
    def __init__(
        self,
        target: str,
        outdir: Optional[Path],
        crf: Optional[int],
        preset: Optional[str],
        two_pass: bool,
        only_codecs: set[str],
    ):
        self.target = target
        self.outdir = outdir
        self.crf = crf
        self.preset = preset
        self.two_pass = two_pass
        self.only_codecs = only_codecs

    def on_created(self, event):
        if isinstance(event, FileCreatedEvent) and not event.is_directory:
            process_if_needed(
                Path(event.src_path), self.target, self.outdir,
                self.crf, self.preset, self.two_pass, self.only_codecs
            )

    def on_moved(self, event):
        if isinstance(event, FileMovedEvent) and not event.is_directory:
            process_if_needed(
                Path(event.dest_path), self.target, self.outdir,
                self.crf, self.preset, self.two_pass, self.only_codecs
            )

# --- CLI ---
def main():
    ap = argparse.ArgumentParser(
        description="Watch a directory and transcode legacy videos (mp4v/mjpeg, etc.) to HEVC (hvc1) or AV1 (av01) in MP4."
    )
    ap.add_argument("watch_dir", nargs="?", default=".", help="Directory to watch (default: current dir)")
    ap.add_argument("--to", choices=["hevc", "av1"], default="hevc", help="Target codec (default: hevc)")
    ap.add_argument("--outdir", type=str, default=None, help="Output directory (default: same as source)")
    ap.add_argument("--crf", type=int, default=None, help="Quality (lower = better)")
    ap.add_argument("--preset", type=str, default=None, help="Speed preset (x265: ultrafast..placebo; SVT-AV1: 0..13; libaom: use digits 0..8)")
    ap.add_argument("--two-pass", action="store_true", help="2-pass for libaom-av1 (ignored for HEVC/SVT-AV1)")
    ap.add_argument("--scan-existing", action="store_true", help="Also process existing files at startup")
    ap.add_argument("--only-codecs", type=str, default=None,
                    help="Comma-separated video codecs to transcode (normalized). Example: 'mpeg4,mjpeg' or 'mp4v,msmpeg4v2'. Default: mp4v,mjpeg.")
    args = ap.parse_args()

    watch_dir = Path(args.watch_dir).resolve()
    outdir = Path(args.outdir).resolve() if args.outdir else None
    only_codecs = parse_codec_list(args.only_codecs)

    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        print("ERROR: ffmpeg and ffprobe must be installed and on PATH.", file=sys.stderr)
        sys.exit(2)

    print(f"[start] Watching: {watch_dir}")
    print(f"[config] target={args.to} outdir={outdir or '(same as source)'} crf={args.crf} preset={args.preset} two_pass={args.two_pass}")
    if only_codecs:
        print(f"[config] only_codecs={sorted(only_codecs)} (normalized)")
    else:
        print(f"[config] only_codecs={sorted(DEFAULT_LEGACY)} (default)")

    if args.scan_existing:
        for p in sorted(watch_dir.iterdir()):
            if p.is_file() and should_handle(p):
                process_if_needed(p, args.to, outdir, args.crf, args.preset, args.two_pass, only_codecs)

    event_handler = Handler(args.to, outdir, args.crf, args.preset, args.two_pass, only_codecs)
    observer = Observer()
    observer.schedule(event_handler, str(watch_dir), recursive=False)
    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[stop] Shutting down...")
        observer.stop()
    observer.join()

if __name__ == "__main__":
    main()