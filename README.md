# Clarity

Local image and video processing — upscale, sharpen, denoise, stabilize.  
No cloud. No subscriptions. Files never leave your machine.

---

## What it does

- **Upscale** — AI upscaling via Real-ESRGAN (2x–4K)
- **Sharpen** — Reduce softness and lens blur (light / medium / heavy)
- **Denoise** — Remove grain and noise (great for low-light footage)
- **Stabilize** — Reduce camera shake in video (ffmpeg vidstab)

Works on photos (jpg, png) and video (mp4, mov, avi, mkv).

---

## Requirements

- Python 3.9+
- ffmpeg
- video2x (for video upscaling)

---

## Setup

### 1. Install ffmpeg

```bash
brew install ffmpeg
```

### 2. Install video2x

```bash
pip install video2x
```

### 3. Clone this repo

```bash
git clone https://github.com/YOUR_USERNAME/clarity.git
cd clarity
```

### 4. Install Python dependencies

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 5. Download Real-ESRGAN model weights (one time)

```bash
mkdir weights
curl -L -o weights/RealESRGAN_x4plus.pth \
  https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth
```

---

## Running

### Start the backend

```bash
source venv/bin/activate
python server.py
```

You'll see:
```
🎞  Clarity backend running at http://localhost:5050
```

### Open the PWA

Visit your GitHub Pages URL (e.g. `https://YOUR_USERNAME.github.io/clarity`) or open `index.html` directly in your browser.

The green dot in the top right confirms the backend is connected.

---

## Installing as a PWA

In Chrome or Edge, click the install icon in the address bar.  
On Safari (Mac), go to File → Add to Dock.

---

## GitHub Pages setup

1. Push this repo to GitHub
2. Go to Settings → Pages
3. Set source to `main` branch, `/ (root)` folder
4. Your PWA will be live at `https://YOUR_USERNAME.github.io/clarity`

The frontend is hosted on GitHub Pages. The backend (`server.py`) always runs locally on your machine — nothing is processed in the cloud.

---

## Notes

- Video processing is slow on CPU — expect 10–30 min per minute of video
- Apple Silicon (M1/M2/M3) will be faster via MPS acceleration
- Always use original, unedited source files for best results
- For low-light footage: enable Denoise before Upscale
