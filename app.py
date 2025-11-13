import streamlit as st
import os
import subprocess
import glob
import pandas as pd
from PIL import Image

# ================== CONFIG ==================
YOLO_DIR = "C:/Users/LENOVO/PycharmProjects/yolov5"
MODEL_PATH = "runs/train/vehicle_model/weights/best.pt"
UPLOAD_DIR = "uploads"
RESULT_DIR = os.path.join(YOLO_DIR, "runs", "detect")
CONF_THRESHOLD = 0.2  # Chỉ tính detection ≥ 0.5
# ============================================

st.set_page_config(page_title="Vehicle Recognition (YOLOv5)", layout="wide")
st.title("🚗 Vehicle Recognition (YOLOv5)")

# Tạo thư mục upload
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Upload ảnh
uploaded_file = st.file_uploader("📸 Chọn ảnh để nhận diện", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
    with open(img_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    col1, col2 = st.columns(2)
    with col1:
        st.image(Image.open(img_path), caption="Ảnh tải lên", width='stretch')

    if st.button("🚀 Nhận diện xe"):
        st.info("Đang chạy YOLOv5 detect.py, vui lòng chờ...")

        # --- Chạy YOLOv5 detect ---
        cmd = [
            "python", os.path.join(YOLO_DIR, "detect.py"),
            "--weights", MODEL_PATH,
            "--img", "640",
            "--source", img_path,
            "--conf", "0.25",
            "--device", "cpu",
            "--save-txt",
            "--save-conf"
        ]

        result = subprocess.run(cmd, cwd=YOLO_DIR, shell=True, capture_output=True, text=True)

        if result.returncode != 0:
            st.error("❌ Lỗi YOLOv5 detect")
            st.code(result.stderr)
        else:
            st.success("✅ YOLOv5 detect hoàn tất!")

            # --- Lấy thư mục exp mới nhất ---
            exp_dirs = sorted(glob.glob(os.path.join(RESULT_DIR, "exp*")), key=os.path.getmtime)
            if not exp_dirs:
                st.error("Không tìm thấy thư mục kết quả (exp)")
            else:
                latest_exp = exp_dirs[-1]

                # Ảnh kết quả
                detected_images = glob.glob(os.path.join(latest_exp, "*.jpg"))
                with col2:
                    if detected_images:
                        st.image(detected_images[0], caption="Ảnh kết quả", width='stretch')
                    else:
                        st.warning("Không tìm thấy ảnh kết quả.")

                # File nhãn
                label_files = glob.glob(os.path.join(latest_exp, "labels", "*.txt"))
                if label_files:
                    label_file = label_files[0]
                    with open(label_file, "r") as f:
                        lines = f.readlines()

                    results = []
                    for line in lines:
                        parts = line.strip().split()
                        if len(parts) >= 6:
                            class_id = int(parts[0])
                            conf = float(parts[-1])
                            if conf >= CONF_THRESHOLD:
                                results.append((class_id, conf))

                    class_names = {0: "Car", 1: "Truck", 2: "Bus", 3: "Motorbike"}

                    if results:
                        df_results = pd.DataFrame(results, columns=["class_id", "conf"])
                        df_results["class_name"] = df_results["class_id"].map(class_names)

                        # Số lượng mỗi loại
                        count_df = df_results["class_name"].value_counts().reset_index()
                        count_df.columns = ["Loại xe", "Số lượng"]

                        # Độ tin cậy trung bình
                        mean_conf = df_results.groupby("class_name")["conf"].mean() * 100
                        mean_conf = mean_conf.round(2)

                        # --- Hiển thị kết quả ---
                        st.subheader("📊 Thống kê phương tiện (conf ≥ 0.5)")
                        col3, col4 = st.columns(2)
                        with col3:
                            st.write("Số lượng từng loại:")
                            st.dataframe(count_df, use_container_width=True)
                        with col4:
                            st.write("Độ tin cậy trung bình:")
                            for cname, cval in mean_conf.items():
                                st.metric(cname, f"{cval}%")

                        # Xe phổ biến nhất
                        top = count_df.iloc[0]
                        st.success(f"🚘 Xe phổ biến nhất: **{top['Loại xe']}** – {top['Số lượng']} chiếc")
                    else:
                        st.warning(f"Không phát hiện được phương tiện nào với conf ≥ {CONF_THRESHOLD}")
                else:
                    st.error("Không tìm thấy xe trong hình.")
