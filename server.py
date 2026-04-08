#!/usr/bin/env python3
"""
Clarity — local processing backend
Run with: python server.py
"""

import os
import re
import subprocess
import tempfile
import threading
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

jobs = {}  # job_id -> { status, progress, message, output_path }


def run_command(cmd, job_id, label="Processing"):
    """Run a shell command and stream progress to job state."""
    log.info(f"[{job_id}] Running: {' '.join(cmd)}")
    jobs[job_id]["message"] = label
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        for line in proc.stdout:
            line = line.strip()
            if line:
                log.info(f"[{job_id}] {line}")
                # Try to parse frame progress from video2x or ffmpeg
                if "frame" in line.lower() or "%" in line:
                    jobs[job_id]["message"] = line[:80]
        proc.wait()
        return proc.returncode == 0
    except Exception as e:
        log.error(f"[{job_id}] Command failed: {e}")
        return False


def process_video(job_id, input_path, output_path, options):
    """Full video processing pipeline."""
    try:
        jobs[job_id]["status"] = "running"
        jobs[job_id]["progress"] = 5

        current = input_path
        step_files = []

        # ── Step 1: Denoise (ffmpeg hqdn3d) ─────────────────────────────────
        if options.get("denoise"):
            strength = options.get("denoise_strength", "medium")
            luma = {"light": "2:1.5:3:2.5", "medium": "4:3:6:4.5", "heavy": "6:5:9:6.5"}[strength]
            denoised = str(input_path).replace(".mov", "_dn.mov").replace(".mp4", "_dn.mp4")
            denoised = str(OUTPUT_DIR / f"{job_id}_denoised.mp4")
            ok = run_command([
                "ffmpeg", "-y", "-i", str(current),
                "-vf", f"hqdn3d={luma}",
                "-c:a", "copy", denoised
            ], job_id, "Denoising...")
            if not ok:
                raise RuntimeError("Denoise step failed")
            step_files.append(denoised)
            current = denoised
            jobs[job_id]["progress"] = 30

        # ── Step 2: Upscale (video2x) ─────────────────────────────────────
        if options.get("upscale"):
            target_h = options.get("target_height", 1080)
            model = options.get("model", "realesr-general-x4v3")
            upscaled = str(OUTPUT_DIR / f"{job_id}_upscaled.mp4")
            ok = run_command([
                "video2x", "-i", str(current), "-o", upscaled,
                "-h", str(target_h), "upscale", "--model", model
            ], job_id, f"Upscaling to {target_h}p...")
            if not ok:
                raise RuntimeError("Upscale step failed")
            step_files.append(upscaled)
            current = upscaled
            jobs[job_id]["progress"] = 75

        # ── Step 3: Sharpen (ffmpeg unsharp) ──────────────────────────────
        if options.get("sharpen"):
            strength = options.get("sharpen_strength", "medium")
            # unsharp=luma_msize_x:luma_msize_y:luma_amount
            amounts = {"light": "3:3:0.5", "medium": "5:5:1.0", "heavy": "7:7:1.8"}[strength]
            sharpened = str(OUTPUT_DIR / f"{job_id}_sharpened.mp4")
            ok = run_command([
                "ffmpeg", "-y", "-i", str(current),
                "-vf", f"unsharp={amounts}",
                "-c:a", "copy", sharpened
            ], job_id, "Sharpening...")
            if not ok:
                raise RuntimeError("Sharpen step failed")
            step_files.append(sharpened)
            current = sharpened
            jobs[job_id]["progress"] = 90

        # ── Step 4: Stabilize (ffmpeg vidstab) ────────────────────────────
        if options.get("stabilize"):
            transforms = str(OUTPUT_DIR / f"{job_id}_transforms.trf")
            stabilized = str(OUTPUT_DIR / f"{job_id}_stabilized.mp4")
            # Pass 1 — detect
            run_command([
                "ffmpeg", "-y", "-i", str(current),
                "-vf", f"vidstabdetect=result={transforms}",
                "-f", "null", "-"
            ], job_id, "Analyzing motion...")
            # Pass 2 — stabilize
            ok = run_command([
                "ffmpeg", "-y", "-i", str(current),
                "-vf", f"vidstabtransform=input={transforms}:smoothing=10",
                "-c:a", "copy", stabilized
            ], job_id, "Stabilizing...")
            if ok:
                step_files.append(stabilized)
                current = stabilized

        # ── Final: copy to output path ─────────────────────────────────────
        import shutil
        shutil.copy2(current, str(output_path))

        # Clean up intermediate files
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
        log.error(f"[{job_id}] Error: {e}")
        jobs[job_id]["status"] = "error"
        jobs[job_id]["message"] = str(e)


def process_photo(job_id, input_path, output_path, options):
    """Photo processing pipeline."""
    try:
        jobs[job_id]["status"] = "running"
        jobs[job_id]["progress"] = 10

        current = str(input_path)

        # ── Upscale with Real-ESRGAN ───────────────────────────────────────
        if options.get("upscale"):
            from realesrgan import RealESRGANer
            from basicsr.archs.rrdbnet_arch import RRDBNet
            import cv2
            import numpy as np

            jobs[job_id]["message"] = "Upscaling..."
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                            num_block=23, num_grow_ch=32, scale=4)
            upsampler = RealESRGANer(
                scale=4,
                model_path="weights/RealESRGAN_x4plus.pth",
                model=model,
                tile=0,
                tile_pad=10,
                pre_pad=0,
                half=False,
            )
            img = cv2.imread(current, cv2.IMREAD_UNCHANGED)
            output, _ = upsampler.enhance(img, outscale=options.get("scale", 4))
            upscaled_path = str(OUTPUT_DIR / f"{job_id}_up.png")
            cv2.imwrite(upscaled_path, output)
            current = upscaled_path
            jobs[job_id]["progress"] = 60

        # ── Sharpen with ffmpeg (works on images too via lavfi) ────────────
        if options.get("sharpen"):
            strength = options.get("sharpen_strength", "medium")
            amounts = {"light": "3:3:0.5", "medium": "5:5:1.0", "heavy": "7:7:1.8"}[strength]
            sharpened = str(OUTPUT_DIR / f"{job_id}_sharp.png")
            run_command([
                "ffmpeg", "-y", "-i", current,
                "-vf", f"unsharp={amounts}",
                sharpened
            ], job_id, "Sharpening...")
            current = sharpened
            jobs[job_id]["progress"] = 85

        import shutil
        shutil.copy2(current, str(output_path))

        jobs[job_id]["status"] = "done"
        jobs[job_id]["progress"] = 100
        jobs[job_id]["message"] = "Complete"
        jobs[job_id]["output_path"] = str(output_path)

    except Exception as e:
        log.error(f"[{job_id}] Error: {e}")
        jobs[job_id]["status"] = "error"
        jobs[job_id]["message"] = str(e)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "version": "1.0.0"})


@app.route("/api/process", methods=["POST"])
def process():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    options = request.form.to_dict()

    # Parse booleans and ints from form data
    for key in ["upscale", "denoise", "sharpen", "stabilize"]:
        options[key] = options.get(key, "false").lower() == "true"
    if "target_height" in options:
        options["target_height"] = int(options["target_height"])
    if "scale" in options:
        options["scale"] = int(options["scale"])

    # Save upload
    job_id = str(uuid.uuid4())[:8]
    suffix = Path(file.filename).suffix.lower()
    input_path = UPLOAD_DIR / f"{job_id}_input{suffix}"
    output_path = OUTPUT_DIR / f"{job_id}_output{suffix}"
    file.save(str(input_path))

    jobs[job_id] = {"status": "queued", "progress": 0, "message": "Queued"}

    is_video = suffix in {".mp4", ".mov", ".avi", ".mkv", ".m4v"}
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
    print("\n🎞  Clarity backend running at http://localhost:5050\n")
    app.run(host="127.0.0.1", port=5050, debug=False)
