"""
web_app.py
==========
FastAPI web app for real-time arecanut ripeness detection using YOLO + DINOv2.

Usage:
    python web_app.py
    Then open: http://localhost:8000
"""

import sys
import time
import threading
from pathlib import Path

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import uvicorn
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from torchvision import transforms
from PIL import Image

try:
    from ultralytics import YOLO
except ImportError:
    print("ultralytics not installed. Please run: pip install ultralytics")
    sys.exit(1)

sys.path.insert(0, str(Path(__file__).parent))
from utils.common import load_config, get_device, setup_logger

# ---------------------------------------------------------------------------
# App Setup
# ---------------------------------------------------------------------------

app = FastAPI(title="Arecanut Ripeness Detection")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

logger = setup_logger("arecanut.web")
config = load_config("config.yaml")
device = get_device(config.detection.device)
class_names = config.classes.names

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ---------------------------------------------------------------------------
# Model Loading (YOLO + DINOv2)
# ---------------------------------------------------------------------------

EMBED_DIMS = {
    "dinov2_vits14": 384,
    "dinov2_vitb14": 768,
    "dinov2_vitl14": 1024,
    "dinov2_vitg14": 1536,
}

dino_model = None
yolo_model = None
model_status = "Loading models..."

class FullModel(nn.Module):
    def __init__(self, backbone, embed_dim, hidden_dim, num_classes, dropout):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.head(self.backbone(x))

def load_models():
    global dino_model, yolo_model, model_status
    
    # 1. Load Custom YOLO Model for Arecanuts
    yolo_path = Path("models/final/yolo_arecanut.pt")
    if not yolo_path.exists():
        model_status = f"ERROR: YOLO model not found at {yolo_path}."
        logger.error(model_status)
        return
        
    try:
        model_status = "Loading YOLO detector..."
        logger.info(model_status)
        yolo_model = YOLO(str(yolo_path))
        yolo_model.to(device)
    except Exception as e:
        model_status = f"ERROR loading YOLO: {e}"
        logger.error(model_status)
        return

    # 2. Load DINOv2 Classifier Model
    ckpt_path = Path(config.paths.checkpoints_dir) / "best_model.pth"
    if not ckpt_path.exists():
        model_status = f"ERROR: No DINO checkpoint found at {ckpt_path}. Run train.py first."
        logger.error(model_status)
        return

    try:
        model_status = "Reading DINOv2 checkpoint..."
        logger.info(model_status)
        state = torch.load(ckpt_path, map_location=device)

        backbone_name = state.get("backbone_name", config.classification.backbone)
        num_classes = state.get("num_classes", config.classes.num_classes)
        embed_dim = EMBED_DIMS.get(backbone_name, 384)

        model_status = f"Building DINOv2 backbone ({backbone_name})..."
        logger.info(model_status)

        backbone = torch.hub.load(
            "facebookresearch/dinov2", backbone_name,
            pretrained=False, verbose=False
        )

        m = FullModel(
            backbone=backbone,
            embed_dim=embed_dim,
            hidden_dim=config.classification.head.hidden_dim,
            num_classes=num_classes,
            dropout=config.classification.head.dropout,
        )

        model_status = "Loading saved weights..."
        logger.info(model_status)
        
        saved_dict = state["model_state_dict"]
        remapped = {}
        for k, v in saved_dict.items():
            new_k = k.replace("head.head.", "head.")
            remapped[new_k] = v

        m.load_state_dict(remapped)
        m = m.to(device).eval()

        dino_model = m
        val_acc = state.get("metrics", {}).get("val_acc", 0)
        model_status = f"Ready ✓  val_acc={val_acc:.1%}"
        logger.info(f"Models loaded successfully. {model_status}")

    except Exception as e:
        model_status = f"ERROR loading DINO: {e}"
        logger.error(model_status, exc_info=True)


threading.Thread(target=load_models, daemon=True).start()

# ---------------------------------------------------------------------------
# Inference Pipeline (YOLO -> DINOv2)
# ---------------------------------------------------------------------------

def classify_crop(crop_bgr: np.ndarray) -> tuple[str, float]:
    """Classify a cropped region using DINOv2."""
    if dino_model is None or crop_bgr.size == 0:
        return "unknown", 0.0
        
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb)
    tensor = transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        probs = F.softmax(dino_model(tensor), dim=-1)[0]
        
    idx = probs.argmax().item()
    return class_names[idx], probs[idx].item()


def annotate_frame(frame: np.ndarray):
    """Detect arecanuts, classify them, draw boxes, return counts."""
    detected = 0
    ripe_count = 0
    unripe_count = 0
    
    if yolo_model is None or dino_model is None:
        cv2.putText(frame, "Model loading...",
                    (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 0), 2)
        return frame, detected, ripe_count, unripe_count

    # Run YOLO detection with a lower confidence threshold for better recall
    results = yolo_model(frame, conf=0.15, verbose=False)
    
    for box in results[0].boxes:
        # Get coordinates
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = box.conf[0].item()
        
        detected += 1
        
        # Crop and classify
        crop = frame[max(0, y1):min(frame.shape[0], y2), max(0, x1):min(frame.shape[1], x2)]
        cls_name, cls_conf = classify_crop(crop)
            
        if cls_name == "ripe":
            ripe_count += 1
            color = (0, 220, 80)  # Green
        else:
            unripe_count += 1
            color = (0, 60, 220)  # Red
            
        # Draw bounding box and label
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{cls_name.upper()} {cls_conf:.0%}"
        cv2.putText(frame, label, (x1, max(y1 - 10, 0)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return frame, detected, ripe_count, unripe_count

# ---------------------------------------------------------------------------
# Live State
# ---------------------------------------------------------------------------

class State:
    fps: float = 0.0
    detected: int = 0
    ripe: int = 0
    unripe: int = 0
    camera_running: bool = False

state = State()

# ---------------------------------------------------------------------------
# Video Stream
# ---------------------------------------------------------------------------

def generate_stream():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("Cannot open webcam.")
        return

    state.camera_running = True
    t_prev = time.time()
    try:
        while state.camera_running:
            ret, frame = cap.read()
            if not ret:
                break
                
            annotated, det, ripe, unripe = annotate_frame(frame)
            
            now = time.time()
            state.fps = round(1.0 / max(now - t_prev, 1e-6), 1)
            t_prev = now
            
            state.detected = det
            state.ripe = ripe
            state.unripe = unripe
            
            _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")
    finally:
        cap.release()
        state.camera_running = False

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(generate_stream(),
                             media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/metrics")
async def get_metrics():
    return {
        "fps": state.fps,
        "detected": state.detected,
        "ripe": state.ripe,
        "unripe": state.unripe,
        "camera_running": state.camera_running,
        "model_ready": (dino_model is not None and yolo_model is not None),
    }


@app.get("/status")
async def get_status():
    """Check model loading progress."""
    return {"status": model_status, "model_ready": (dino_model is not None and yolo_model is not None)}


@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    """Upload an image for ripeness prediction."""
    if dino_model is None or yolo_model is None:
        return JSONResponse({"error": f"Model not ready: {model_status}"}, status_code=503)
        
    contents = await file.read()
    arr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    
    if frame is None:
        return JSONResponse({"error": "Invalid image file"}, status_code=400)
        
    import base64
    annotated, det, ripe, unripe = annotate_frame(frame)
    _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
    b64_str = base64.b64encode(buf).decode('utf-8')
    
    return {"detected": det, "ripe": ripe, "unripe": unripe, "image_b64": b64_str}


if __name__ == "__main__":
    uvicorn.run("web_app:app", host="0.0.0.0", port=8000, reload=False)
