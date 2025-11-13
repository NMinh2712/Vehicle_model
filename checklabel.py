import pandas as pd
import os
import matplotlib.pyplot as plt

# --- Thay đổi theo project ---
project_dir = "runs/train/vehicle_model"
results_csv = os.path.join(project_dir, "results.csv")

# Kiểm tra file có tồn tại
if not os.path.exists(results_csv):
    raise FileNotFoundError(f"Không tìm thấy file: {results_csv}")

# Đọc dữ liệu
df = pd.read_csv(results_csv)

# Strip all column names
df.columns = df.columns.str.strip()

print("=== Columns after stripping ===")
print(df.columns.tolist())

# Chọn các cột quan trọng (nếu tồn tại)
desired_columns = ["epoch", "train/box_loss", "train/obj_loss", "train/cls_loss",
                   "metrics/precision", "metrics/recall", "metrics/mAP_0.5", "metrics/mAP_0.5:0.95",
                   "val/box_loss", "val/obj_loss", "val/cls_loss"]
available_columns = [col for col in desired_columns if col in df.columns]

print("\n=== Available metrics ===")
print(df[available_columns].head())

# Thống kê cuối cùng
print("\n=== Last epoch ===")
print(df[available_columns].iloc[-1])

# Vẽ biểu đồ
plt.figure(figsize=(12,5))

# Loss
plt.subplot(1,2,1)
for col in ["train/box_loss", "train/obj_loss", "train/cls_loss",
            "val/box_loss", "val/obj_loss", "val/cls_loss"]:
    if col in available_columns:
        plt.plot(df["epoch"], df[col], label=col)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("YOLOv5 Training & Validation Loss")
plt.legend()

# Accuracy metrics
plt.subplot(1,2,2)
for col in ["metrics/precision", "metrics/recall", "metrics/mAP_0.5", "metrics/mAP_0.5:0.95"]:
    if col in available_columns:
        plt.plot(df["epoch"], df[col], label=col)
plt.xlabel("Epoch")
plt.ylabel("Score")
plt.title("YOLOv5 Accuracy Metrics")
plt.legend()

plt.tight_layout()
plt.show()
