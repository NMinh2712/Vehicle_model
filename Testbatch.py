import os, glob
from PIL import Image

DATASET_DIR = "/content/dataset"
missing_labels, corrupted_images, invalid_labels = [], [], []

for img_path in glob.glob(f"{DATASET_DIR}/images/**/*.jpg", recursive=True):
    base = os.path.splitext(os.path.basename(img_path))[0]
    lbl_path = f"{DATASET_DIR}/labels/{base}.txt"
    if not os.path.exists(lbl_path):
        missing_labels.append(lbl_path)
    try:
        Image.open(img_path).verify()
    except:
        corrupted_images.append(img_path)

for lbl_path in glob.glob(f"{DATASET_DIR}/labels/**/*.txt", recursive=True):
    with open(lbl_path) as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                cid = int(float(parts[0]))
                if cid not in [0, 1, 2, 3]:
                    invalid_labels.append(lbl_path)
print("Missing labels:", len(missing_labels))
print("Corrupted images:", len(corrupted_images))
print("Invalid labels:", len(invalid_labels))
import glob, os

violations = []
for p in glob.glob(f"{DATASET_DIR}/labels/**/*.txt", recursive=True):
    with open(p) as f:
        for i, line in enumerate(f,1):
            parts = list(map(float, line.strip().split()))
            if len(parts)!=5: continue
            _, x, y, w, h = parts
            if not (0<=x<=1 and 0<=y<=1 and 0<=w<=1 and 0<=h<=1):
                violations.append((p,i,(x,y,w,h)))
print("Normalization violations:", len(violations))