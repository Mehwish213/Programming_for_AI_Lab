# FaceProfile — Facial Analysis & Personality System

A Flask-based web application that detects and measures facial features using **MediaPipe Face Mesh** (468 landmarks), then derives a **MBTI-style 16 personality profile** based on facial geometry.

## Features

- **468-point face landmark detection** via MediaPipe (more precise than dlib's 68 points)
- **Feature measurements:**
  - Eyes (width, height, openness, symmetry, aspect ratio, interocular distance)
  - Nose (width, height, nasal index, shape classification)
  - Mouth (width, height, lip fullness, width ratio)
  - Jawline (width, face shape: Round/Oval/Square/Heart/Diamond/Oblong/Pear/Rectangle)
  - Eyebrows (style: arched/flat, prominence, spacing)
  - Facial thirds balance & golden ratio harmony score
- **Annotated image output** with color-coded landmark overlays
- **MBTI 16 Personality Type profiling** with axis scores, strengths, and growth areas
- **Camera capture** or image upload
- Dark-mode UI with distinctive editorial aesthetic

## Installation

```bash
pip install flask opencv-python mediapipe numpy Pillow
```

## Running

```bash
cd face_profiler
python app.py
```

Then open: **http://localhost:5000**

## How It Works

1. User uploads or captures a front-facing photo
2. MediaPipe Face Mesh detects 468 facial landmarks
3. Key distances are measured between landmark points (in pixels, proportional)
4. Facial feature categories are classified (eye openness, nose shape, face shape, etc.)
5. A personality scoring algorithm maps feature traits to MBTI axes:
   - **E/I**: Eye openness, mouth width, facial symmetry
   - **S/N**: Interocular distance, nose shape, face shape
   - **T/F**: Face shape, lip fullness, brow style
   - **J/P**: Golden ratio harmony, brow prominence, face shape
6. Results shown with annotated image, personality profile, and measurement grid

## Notes

- All processing is local — no images are sent to external servers
- Personality profiling is **for entertainment/educational purposes only**
- Works best with clear, front-facing, well-lit photos
- MediaPipe requires minimum detection confidence of 0.5

## Landmark Color Legend

| Color | Feature |
|-------|---------|
| 🟢 Cyan | Eyes |
| 🟡 Yellow | Nose |
| 🔵 Blue | Mouth |
| 🟢 Green | Jawline |
| 🟣 Purple | Eyebrows |
