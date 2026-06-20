# 🌿 Arecanut Ripeness Detection (YOLO + DINOv2)

A fast, accurate, and real-time computer vision application that detects arecanuts using a webcam or uploaded images, and automatically classifies them as **Ripe** or **Unripe**.

---

## 🧠 How It Works (Two-Stage Pipeline)

This project uses a highly accurate, two-stage AI pipeline:
1. **Detection (YOLO):** A custom-trained YOLO model (`yolo_arecanut.pt`) scans the live video feed or image and draws bounding boxes specifically around arecanuts, ignoring backgrounds and other objects (like faces or hands).
2. **Classification (DINOv2):** Every bounding box found by YOLO is cropped and passed to Meta's DINOv2 Vision Transformer. DINOv2 acts as a powerful feature extractor and binary classifier to accurately determine if the specific arecanut inside that box is `ripe` or `unripe`.

---

## 📂 Project Structure

```text
arecanut_ripeness/
├── models/
│   ├── final/
│   │   └── yolo_arecanut.pt       # Custom YOLO Object Detector weights
│   └── checkpoints/
│       └── best_model.pth         # Trained DINOv2 Classifier weights
├── dataset/                       # Training images
│   ├── train/ripe/ & unripe/
│   ├── val/ripe/ & unripe/
│   └── test/ripe/ & unripe/
├── static/                        # Web App UI Assets
│   ├── style.css                  # Modern Glassmorphism CSS
│   └── script.js                  # Frontend logic & API polling
├── templates/
│   └── index.html                 # Main Web Interface
├── utils/
│   ├── common.py                  # Config, logging, device selection
│   ├── dataset.py                 # DataLoaders for PyTorch
│   └── model.py                   # DINOv2 model architecture definition
├── config.yaml                    # Master configuration file
├── train.py                       # Script to train the DINOv2 classifier
├── web_app.py                     # FastAPI server handling live video & inference
└── requirements.txt               # Required Python packages
```

---

## 🚀 Getting Started

### 1. Install Dependencies
Make sure you have Python installed, then install the required libraries:
```bash
pip install -r requirements.txt
```

### 2. Run the Web Application
Start the FastAPI server to launch the real-time detection UI:
```bash
python web_app.py
```
* Once it says `Uvicorn running`, open your web browser and navigate to: **http://localhost:8000**
* **Live Feed:** The app will automatically connect to your webcam and display real-time detection and counting metrics.
* **Upload Image:** You can scroll down on the right sidebar to upload a static photo and see the bounding boxes drawn over it instantly.

---

## 🛠️ Retraining the Classifier

If you gather new images and want to improve the Ripe/Unripe accuracy, you can retrain the DINOv2 classifier model.

1. Place your new images into the `dataset/train/ripe` and `dataset/train/unripe` folders.
2. Run the training script:
```bash
python train.py
```
*(You can also adjust parameters like `--epochs 30` or change the learning rate in `config.yaml`)*

The script will automatically freeze the DINOv2 backbone for the first few epochs (to quickly train the classification head), and then unfreeze it to fine-tune the entire network. The best model will be automatically saved to `models/checkpoints/best_model.pth`.

---

## ⚙️ Configuration (`config.yaml`)

You can edit `config.yaml` to easily change core parameters without touching the code:
- **`classification.backbone`**: Change the DINOv2 model size (`dinov2_vits14`, `vitb14`, or `vitl14`).
- **`classification.training.epochs`**: Total number of epochs for training.
- **`detection.device`**: Force the app to run on `"cpu"`, `"cuda"` (GPU), or `"auto"`.
