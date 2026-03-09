import streamlit as st
import os
os.environ['YOLO_CONFIG_DIR'] = '/tmp'
from ultralytics import YOLO
from PIL import Image
import numpy as np

MODEL_PATH = "my_model_v8/my_model_v8.pt"
model = YOLO(MODEL_PATH)

st.title("🐝 Beehive Cell Detection with YOLOv8")

uploaded_file = st.file_uploader("Upload a beehive image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    img_array = np.array(image)

    results = model.predict(img_array, device="cpu")

    st.subheader("Detections:")
    for result in results:
        # Show detection image for each result
        result_img = result.plot()
        if result_img is not None and result_img.shape[0] > 0 and result_img.shape[1] > 0:
            st.image(result_img, caption="Detection Results", use_container_width=True)

        # Show detection details for each result
        for box in result.boxes:
            cls_id = int(box.cls.cpu().numpy().item())
            conf = float(box.conf.cpu().numpy().item())
            st.write(f"Class: {model.names[cls_id]} | Confidence: {conf:.2f}")
