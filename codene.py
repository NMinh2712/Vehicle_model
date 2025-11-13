import yaml
import os

# Thư mục gốc dataset vehicle
root = r"D:/2025/FA25/AIL/yolo"

# Tên file YAML sẽ tạo
yaml_file = os.path.join(root, "vehicle.yaml")

# Class của bạn
names = ["car", "truck", "bus", "motorcycle"]

# Dữ liệu train/val
data = {
    "train": f"{root}/images/train",  # thư mục train của vehicle
    "val": f"{root}/images/val",      # thư mục val của vehicle
    "nc": len(names),
    "names": names
}

# Ghi file YAML
with open(yaml_file, "w", encoding="utf-8") as f:
    yaml.dump(data, f, default_flow_style=False, sort_keys=False)

print(f"✅ Created: {yaml_file}")
