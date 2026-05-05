# Real-Time Face Detection with OpenCV

A lightweight Python script that detects human faces in real time using your webcam and OpenCV's Haar Cascade classifier.

---

## Demo

When running, the program opens a live camera window and draws a bounding box with a **"Face Detected"** label around every face it finds.

---

## Requirements

- Python 3.x
- OpenCV

Install OpenCV via pip:

```bash
pip install opencv-python
```

---

## Setup

### 1. Get the Haar Cascade file

Download the pre-trained face classifier XML file from the official OpenCV GitHub repository:

[haarcascade_frontalface_default.xml](https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml)

Save it somewhere on your machine, for example:

```
C:/Users/YourName/models/haarcascade_frontalface_default.xml   # Windows
/home/yourname/models/haarcascade_frontalface_default.xml      # Linux/macOS
```

### 2. Update the file path in the script

Open `face_detector.py` and replace the empty string with the actual path to your XML file:

```python
# Before
face_cascade_path = " "

# After
face_cascade_path = "C:/Users/YourName/models/haarcascade_frontalface_default.xml"
```

### 3. Set the correct camera index

```python
cap = cv2.VideoCapture(1)  # Change 1 to 0 if using your default/built-in webcam
```

| Value | Camera |
|-------|--------|
| `0`   | Default / built-in webcam |
| `1`   | External / secondary webcam |

---

## Usage

Run the script from your terminal:

```bash
python face_detector.py
```

A window titled **"Face Detector"** will open showing the live camera feed.

Press **`Q`** to quit.

---

## How It Works

| Step | What happens |
|------|--------------|
| Load classifier | Reads the Haar Cascade XML file with pre-trained face patterns |
| Capture frame | Reads one frame at a time from the webcam |
| Grayscale conversion | Converts the frame to grayscale (required by the detector) |
| Face detection | Scans the frame using `detectMultiScale()` with configurable sensitivity |
| Draw results | Draws a blue rectangle and a yellow label over each detected face |
| Display | Shows the annotated frame in a live window |

### Detection Parameters

```python
face_cascade.detectMultiScale(grey, scaleFactor=1.5, minNeighbors=5)
```

| Parameter | Value | Effect |
|-----------|-------|--------|
| `scaleFactor` | `1.5` | How much the image is scaled down at each step — higher = faster but may miss small faces |
| `minNeighbors` | `5` | Minimum detections needed to confirm a face — higher = fewer false positives |

---

## Project Structure

```
face-detector/
│
├── face_detector.py                      # Main script
├── haarcascade_frontalface_default.xml   # Haar Cascade model (download separately)
└── README.md
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| Black screen / no window | Check camera index (`0` or `1`) |
| No faces detected | Verify the XML path is correct and the file exists |
| Script crashes immediately | Make sure `face_cascade_path` is not empty |
| Too many false positives | Increase `minNeighbors` (e.g., `7` or `10`) |
| Misses real faces | Decrease `scaleFactor` (e.g., `1.1`) or `minNeighbors` |

---

## License

This project is open source and free to use for learning and personal projects.
