from pathlib import Path

from models.common import DetectMultiBackend

weights = Path(r"C:/Users/LENOVO/PycharmProjects/yolov5/runs/train/vehicle_model_new/weights/best.pt")
device = "cuda:0"  # hoặc "cpu"

model = DetectMultiBackend(weights, device=device)
