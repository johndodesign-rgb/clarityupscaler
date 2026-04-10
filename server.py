#!/usr/bin/env python3
"""
Clarity — local processing backend
Run with: caffeinate -i python server.py
"""

import os
import json
import shutil
import subprocess
import tempfile
import threading
import traceback
import uuid
from pathlib import Path
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
import logging

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("clarity")

app = Flask(__name__)
CORS(app, origins=["https://*.github.io", "http://localhost:*", "http://127.0.0.1:*"])

UPLOAD_DIR = Path(tempfile.gettempdir()) / "clarity_uploads"
OUTPUT_DIR = Path(tempfile.gettempdir()) / "clarity_outputs"
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

FFMPEG  = "/opt/homebrew/bin/ffmpeg"
FFPROBE = "/opt/homebrew/bin/ffprobe"
WEIGHTS = str(Path(__file__).parent / "weights" / "RealESRGAN_x4plus.pth")

jobs = {}


def run_command(cmd, job_id, label="Processing"):
    log.info(f"[{job_id}] Running: {' '.join(str(c) for c in cmd)}")
    jobs[job_id]["message"] = label
    try:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        for line in proc.stdout:
            line = line.strip()
            if line:
                log.info(f"[{job_id}] {line}")
                if "frame" in line.lower() or "%" in line:
                    jobs[job_id]["message"] = line[:80]
        proc.wait()
        return proc.returncode == 0
    except Exception as e:
        log.error(f"[{job_id}] Command failed: {e}")
        return False


def get_video_fps(path):
    """Detect original video frame rate."""
    try:
        result = subprocess.run([
            FFPROBE, "-v", "quiet",
            "-select_streams", "v:0",
            "-show_entries", "stream=r_frame_rate",
            "-of", "json", str(path)
        ], capture_output=True, text=True)
        data = json.loads(result.stdout)
        rate = data["streams"][0]["r_frame_rate"]
        num, den = rate.split("/")
        fps = float(num) / float(den)
        log.info(f"Detected FPS: {fps}")
        return fps
    except Exception as e:
        log.warning(f"Could not detect FPS, defaulting to 30: {e}")
        return 30.0


def convert_to_mp4(input_path, job_id):
    """Convert any video format to mp4 for consistent processing."""
    suffix = Path(input_path).suffix.lower()
    if suffix == ".mp4":
        return str(input_path)
    converted = str(OUTPUT_DIR / f"{job_id}_converted.mp4")
    ok = run_command([
        FFMPEG, "-y", "-i", str(input_path),
        "-c:v", "libx264", "-c:a", "aac", converted
    ], job_id, "Converting format...")
    return converted if ok else str(input_path)


def process_video(job_id, input_path, output_path, options):
    try:
        jobs[job_id]["status"] = "running"
        jobs[job_id]["progress"] = 2

        original_fps = get_video_fps(input_path)
        current = convert_to_mp4(input_path, job_id)
        step_files = []

        # ── Denoise ───────────────────────────────────────────────────────
        if options.get("denoise"):
            strength = options.get("denoise_strength", "medium")
            luma = {"light": "2:1.5:3:2.5", "medium": "4:3:6:4.5", "heavy": "6:5:9:6.5"}[strength]
            denoised = str(OUTPUT_DIR / f"{job_id}_denoised.mp4")
            ok = run_command([
                FFMPEG, "-y", "-i", str(current),
                "-vf", f"hqdn3d={luma}", "-c:a", "copy", denoised
            ], job_id, "Denoising...")
            if not ok:
                raise RuntimeError("Denoise step failed")
            step_files.append(denoised)
            current = denoised
            jobs[job_id]["progress"] = 20

        # ── Upscale (Real-ESRGAN frame by frame) ──────────────────────────
        if options.get("upscale"):
            import cv2
            from realesrgan import RealESRGANer
            from basicsr.archs.rrdbnet_arch import RRDBNet

            frames_dir = OUTPUT_DIR / f"{job_id}_frames"
            upscaled_dir = OUTPUT_DIR / f"{job_id}_upscaled_frames"
            frames_dir.mkdir(exist_ok=True)
            upscaled_dir.mkdir(exist_ok=True)

            run_command([
                FFMPEG, "-y", "-i", str(current),
                str(frames_dir / "frame%06d.png")
            ], job_id, "Extracting frames...")

            model_obj = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                                num_block=23, num_grow_ch=32, scale=4)
            upsampler = RealESRGANer(
                scale=4, model_path=WEIGHTS, model=model_obj,
                tile=256, tile_pad=10, pre_pad=0, half=False,
            )

            frames = sorted(frames_dir.glob("*.png"))
            total = len(frames)
            base = 20 if options.get("denoise") else 5

            for i, frame_path in enumerate(frames):
                img = cv2.imread(str(frame_path), cv2.IMREAD_UNCHANGED)
                if img is None:
                    continue
                out, _ = upsampler.enhance(img, outscale=2)
                cv2.imwrite(str(upscaled_dir / frame_path.name), out)
                jobs[job_id]["progress"] = base + int((i / total) * 55)
                jobs[job_id]["message"] = f"Upscaling frame {i+1}/{total}"

            upscaled_mp4 = str(OUTPUT_DIR / f"{job_id}_upscaled.mp4")
            run_command([
                FFMPEG, "-y",
                "-framerate", str(original_fps),
                "-i", str(upscaled_dir / "frame%06d.png"),
                "-i", str(current),
                "-map", "0:v", "-map", "1:a?",
                "-c:v", "libx264", "-crf", "18",
                "-c:a", "aac", upscaled_mp4
            ], job_id, "Reassembling video...")

            step_files.append(upscaled_mp4)
            current = upscaled_mp4
            jobs[job_id]["progress"] = 75

            shutil.rmtree(str(frames_dir), ignore_errors=True)
            shutil.rmtree(str(upscaled_dir), ignore_errors=True)

        # ── Sharpen ───────────────────────────────────────────────────────
        if options.get("sharpen"):
            strength = options.get("sharpen_strength", "medium")
            amounts = {"light": "3:3:0.5", "medium": "5:5:1.0", "heavy": "7:7:1.8"}[strength]
            sharpened = str(OUTPUT_DIR / f"{job_id}_sharpened.mp4")
            ok = run_command([
                FFMPEG, "-y", "-i", str(current),
                "-vf", f"unsharp={amounts}", "-c:a", "copy", sharpened
            ], job_id, "Sharpening...")
            if ok:
                step_files.append(sharpened)
                current = sharpened
            jobs[job_id]["progress"] = 85

        # ── Brightness / Contrast ─────────────────────────────────────────
        if options.get("brightness"):
            brightness = options.get("brightness_amount", "0.1")
            contrast = options.get("contrast_amount", "1.2")
            brightened = str(OUTPUT_DIR / f"{job_id}_bright.mp4")
            ok = run_command([
                FFMPEG, "-y", "-i", str(current),
                "-vf", f"eq=brightness={brightness}:contrast={contrast}",
                "-c:a", "copy", brightened
            ], job_id, "Adjusting brightness...")
            if ok:
                step_files.append(brightened)
                current = brightened
            jobs[job_id]["progress"] = 92

        # ── Stabilize ─────────────────────────────────────────────────────
        if options.get("stabilize"):
            transforms = str(OUTPUT_DIR / f"{job_id}_transforms.trf")
            stabilized = str(OUTPUT_DIR / f"{job_id}_stabilized.mp4")
            run_command([
                FFMPEG, "-y", "-i", str(current),
                "-vf", f"vidstabdetect=result={transforms}",
                "-f", "null", "-"
            ], job_id, "Analyzing motion...")
            ok = run_command([
                FFMPEG, "-y", "-i", str(current),
                "-vf", f"vidstabtransform=input={transforms}:smoothing=10",
                "-c:a", "copy", stabilized
            ], job_id, "Stabilizing...")
            if ok:
                step_files.append(stabilized)
                current = stabilized

        shutil.copy2(current, str(output_path))
        for f in step_files:
            try:
                if f != str(output_path):
                    os.remove(f)
            except Exception:
                pass

        jobs[job_id]["status"] = "done"
        jobs[job_id]["progress"] = 100
        jobs[job_id]["message"] = "Complete"
        jobs[job_id]["output_path"] = str(output_path)

    except Exception as e:
        log.error(f"[{job_id}] Error: {e}\n{traceback.format_exc()}")
        jobs[job_id]["status"] = "error"
        jobs[job_id]["message"] = str(e)


def process_photo(job_id, input_path, output_path, options):
    try:
        jobs[job_id]["status"] = "running"
        jobs[job_id]["progress"] = 10
        current = str(input_path)

        # ── Upscale ───────────────────────────────────────────────────────
        if options.get("upscale"):
            import cv2
            from realesrgan import RealESRGANer
            from basicsr.archs.rrdbnet_arch import RRDBNet

            jobs[job_id]["message"] = "Upscaling..."
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                            num_block=23, num_grow_ch=32, scale=4)
            upsampler = RealESRGANer(
                scale=4, model_path=WEIGHTS, model=model,
                tile=256, tile_pad=10, pre_pad=0, half=False,
            )
            img = cv2.imread(current, cv2.IMREAD_UNCHANGED)
            if img is None:
                raise RuntimeError("Could not read image file")
            output, _ = upsampler.enhance(img, outscale=options.get("scale", 4))
            upscaled_path = str(OUTPUT_DIR / f"{job_id}_up.png")
            cv2.imwrite(upscaled_path, output)
            current = upscaled_path
            jobs[job_id]["progress"] = 60

        # ── Sharpen (Pillow) ──────────────────────────────────────────────
        if options.get("sharpen"):
            from PIL import Image, ImageFilter, ImageEnhance
            strength = options.get("sharpen_strength", "medium")
            amount = {"light": 1.2, "medium": 1.6, "heavy": 2.2}[strength]
            img = Image.open(current)
            for _ in range(int(amount)):
                img = img.filter(ImageFilter.SHARPEN)
            img = ImageEnhance.Sharpness(img).enhance(amount)
            sharpened_path = str(OUTPUT_DIR / f"{job_id}_sharp.png")
            img.save(sharpened_path)
            current = sharpened_path
            jobs[job_id]["progress"] = 80

        # ── Brightness / Contrast (Pillow) ────────────────────────────────
        if options.get("brightness"):
            from PIL import Image, ImageEnhance
            brightness = float(options.get("brightness_amount", "0.1"))
            contrast = float(options.get("contrast_amount", "1.2"))
            img = Image.open(current)
            img = ImageEnhance.Brightness(img).enhance(1.0 + brightness)
            img = ImageEnhance.Contrast(img).enhance(contrast)
            bright_path = str(OUTPUT_DIR / f"{job_id}_bright.png")
            img.save(bright_path)
            current = bright_path
            jobs[job_id]["progress"] = 90

        shutil.copy2(current, str(output_path))
        jobs[job_id]["status"] = "done"
        jobs[job_id]["progress"] = 100
        jobs[job_id]["message"] = "Complete"
        jobs[job_id]["output_path"] = str(output_path)

    except Exception as e:
        log.error(f"[{job_id}] Error: {e}\n{traceback.format_exc()}")
        jobs[job_id]["status"] = "error"
        jobs[job_id]["message"] = str(e)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_file("index.html")

@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "version": "1.1.0"})

@app.route("/api/process", methods=["POST"])
def process():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    options = request.form.to_dict()

    for key in ["upscale", "denoise", "sharpen", "stabilize", "brightness"]:
        options[key] = options.get(key, "false").lower() == "true"
    if "target_height" in options:
        options["target_height"] = int(options["target_height"])
    if "scale" in options:
        options["scale"] = int(options["scale"])

    job_id = str(uuid.uuid4())[:8]
    suffix = Path(file.filename).suffix.lower()
    input_path = UPLOAD_DIR / f"{job_id}_input{suffix}"
    output_path = OUTPUT_DIR / f"{job_id}_output{suffix}"
    file.save(str(input_path))

    jobs[job_id] = {"status": "queued", "progress": 0, "message": "Queued"}

    is_video = suffix in {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".3gp", ".wmv"}
    fn = process_video if is_video else process_photo
    t = threading.Thread(target=fn, args=(job_id, input_path, output_path, options), daemon=True)
    t.start()

    return jsonify({"job_id": job_id})

@app.route("/api/status/<job_id>")
def status(job_id):
    job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    return jsonify(job)

@app.route("/api/download/<job_id>")
def download(job_id):
    job = jobs.get(job_id)
    if not job or job["status"] != "done":
        return jsonify({"error": "Not ready"}), 404
    path = job["output_path"]
    return send_file(path, as_attachment=True,
                     download_name=f"clarity_{job_id}{Path(path).suffix}")


if __name__ == "__main__":
    print("\nClarity backend running at http://127.0.0.1:5050\n")
    app.run(host="127.0.0.1", port=5050, debug=False)
