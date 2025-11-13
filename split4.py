import os, shutil, random, math

root = r"D:/2025/FA25/AIL/yolo"

img_dir = os.path.join(root, "images/train")
lbl_dir = os.path.join(root, "labels/train")

# Lấy danh sách ảnh
imgs = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.png'))])
random.shuffle(imgs)

total = len(imgs)
print("Total images:", total)

# Chia đều thành 4 phần
split_size = math.ceil(total / 4)
parts = [imgs[i:i + split_size] for i in range(0, total, split_size)]

# Đảm bảo chỉ lấy đúng 4 phần
parts = parts[:4]

# Tạo folder part1 → part4
for i in range(1, 5):
    os.makedirs(f"{root}/part{i}/images", exist_ok=True)
    os.makedirs(f"{root}/part{i}/labels", exist_ok=True)

# Copy file theo từng phần
for part_index, part_imgs in enumerate(parts, start=1):
    print(f"Copying part {part_index} - {len(part_imgs)} images")

    for img in part_imgs:
        src_img = os.path.join(img_dir, img)
        dst_img = f"{root}/part{part_index}/images/{img}"

        shutil.copy(src_img, dst_img)

        txt = img.rsplit('.', 1)[0] + ".txt"
        src_lbl = os.path.join(lbl_dir, txt)
        dst_lbl = f"{root}/part{part_index}/labels/{txt}"

        if os.path.exists(src_lbl):
            shutil.copy(src_lbl, dst_lbl)
        else:
            print("WARNING: Label missing for", img)

print("✅ Done splitting dataset into 4 parts.")
