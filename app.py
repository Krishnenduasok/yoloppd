import streamlit as st
from PIL import Image
from ultralytics import YOLO
import os

# --- Class labels (in same order as training) ---
class_names = ['high', 'low', 'md', 'medium', 'zero']

# --- Load YOLOv8 classification model ---
@st.cache_resource
def load_model():
    model = YOLO("cassava_ppd_yolov8.pt")  # Local YOLOv8 .pt file
    return model

model = load_model()

# --- Streamlit UI ---
st.markdown("<h1 style='color:#198754;'>🧪 PPD Score Prediction from Tuber Images of Cassava</h1>", unsafe_allow_html=True)
st.subheader("Upload a cassava tuber image to predict the Postharvest Physiological Deterioration (PPD) score.")

# --- File upload ---
uploaded_file = st.file_uploader("Choose a cassava tuber image...", type=["jpg", "jpeg", "png"])

# --- Sidebar: show example images (if available) ---
st.sidebar.markdown("### 📊 Example Class Images")
example_folder = "examples"
for class_name in class_names:
    class_path = os.path.join(example_folder, class_name)
    if os.path.isdir(class_path):
        st.sidebar.markdown(f"**{class_name.upper()}**")
        image_files = [f for f in os.listdir(class_path) if f.lower().endswith(('jpg', 'jpeg', 'png'))]
        for img_file in image_files[:2]:
            img_path = os.path.join(class_path, img_file)
            st.sidebar.image(img_path, use_container_width=False, width=160)
    else:
        st.sidebar.warning(f"No folder for '{class_name}'")

# --- Prediction section ---
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", width=300)

    results = model.predict(image, imgsz=224)
    predicted_class = class_names[int(results[0].probs.top1)]

    st.success(f"✅ Predicted Class: **{predicted_class.upper()}**")
