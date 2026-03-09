import streamlit as st
import os
os.environ['YOLO_CONFIG_DIR'] = '/tmp'
from ultralytics import YOLO
from PIL import Image
import numpy as np
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload
from google.oauth2 import service_account
import io
import cv2
from datetime import datetime

MODEL_PATH = "my_model_v8/my_model_v8.pt"
model = YOLO(MODEL_PATH)

def get_drive_service():
    creds_dict = dict(st.secrets["google"])
    creds = service_account.Credentials.from_service_account_info(
        creds_dict,
        scopes=["https://www.googleapis.com/auth/drive"]
    )
    return build("drive", "v3", credentials=creds)

def upload_to_drive(service, image_bytes: io.BytesIO, filename: str, folder_id: str):
    try:
        file_metadata = {
            "name": filename,
            "parents": [folder_id]
        }
        media = MediaIoBaseUpload(image_bytes, mimetype="image/jpeg")
        service.files().create(body=file_metadata, media_body=media).execute()
        return True
    except Exception as e:
        st.warning(f"Could not save {filename} to Drive: {e}")
        return False

st.title("🐝 Beehive Cell Detection with YOLOv8")

uploaded_file = st.file_uploader("Upload a beehive image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"{timestamp}_{uploaded_file.name}"

    # Run detection
    img_array = np.array(image)
    results = model.predict(img_array, device="cpu")

    # Get Drive service once
    service = get_drive_service()
    originals_folder = st.secrets["folders"]["originals"]
    detections_folder = st.secrets["folders"]["detections"]

    # Upload original image
    original_buf = io.BytesIO()
    image.save(original_buf, format="JPEG")
    original_buf.seek(0)
    upload_to_drive(service, original_buf, f"original_{base_name}", originals_folder)

    # Show and upload detection results
    st.subheader("Detections:")
    for i, result in enumerate(results):
        result_img = result.plot()  # numpy array (BGR)

        if result_img is not None and result_img.shape[0] > 0 and result_img.shape[1] > 0:
            # Display (convert BGR → RGB for streamlit)
            result_img_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
            st.image(result_img_rgb, caption=f"Detection Results {i+1}", use_container_width=True)

            # Upload detection image to Drive
            detection_pil = Image.fromarray(result_img_rgb)
            detection_buf = io.BytesIO()
            detection_pil.save(detection_buf, format="JPEG")
            detection_buf.seek(0)
            upload_to_drive(service, detection_buf, f"detection_{i+1}_{base_name}", detections_folder)

        # Show detection details
        for box in result.boxes:
            cls_id = int(box.cls.cpu().numpy().item())
            conf = float(box.conf.cpu().numpy().item())
            st.write(f"Class: {model.names[cls_id]} | Confidence: {conf:.2f}")

st.success("✅ Images saved to Google Drive!")
