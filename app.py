import streamlit as st
from PIL import Image
import gdown
import os
import torch
from torchvision import transforms

from model_resnet50 import ResNet50WithDropout
from model_efficientnet import EfficientNetV2SWithDropout
from ultralytics import YOLO

# Class labels
class_names = ['high', 'low', 'md', 'medium', 'zero']

# Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Helper to download if not present
def download_if_needed(file_id, output):
    if not os.path.exists(output):
        gdown.download(f"https://drive.google.com/uc?id={file_id}", output, quiet=False)

@st.cache_resource
def load_models():
    models = {}

    # YOLOv8 (Already local)
    try:
        models['YOLOv8'] = YOLO("cassava_ppd_yolov8.pt")
    except Exception as e:
        st.warning(f"YOLOv8 load failed: {e}")

    # ResNet50
    try:
        resnet_path = "resnetfinal_state_dict.pth"
        download_if_needed("11Kwodly2XNUcOt7HdlBbD77sTsC4_I9o", resnet_path)

        resnet_model = ResNet50WithDropout(num_classes=len(class_names))
        resnet_model.load_state_dict(torch.load(resnet_path, map_location="cpu"))
        resnet_model.eval()
        models['ResNet50'] = resnet_model
    except Exception as e:
        st.warning(f"ResNet50 load failed: {e}")

    # EfficientNet
    try:
        eff_path = "efficientnet_state_dict.pth"
        download_if_needed("10wlsWr-St47LCUQ7wGqH5BecrGPJHJwL-", eff_path)

        eff_model = EfficientNetV2SWithDropout(num_classes=len(class_names))
        eff_model.load_state_dict(torch.load(eff_path, map_location="cpu"))
        eff_model.eval()
        models['EfficientNetV2S'] = eff_model
    except Exception as e:
        st.warning(f"EfficientNet load failed: {e}")

    return models

models = load_models()

# UI
st.markdown("<h1 style='color:#198754;'>🧪 PPD Score Prediction from Tuber Images of Cassava</h1>", unsafe_allow_html=True)
st.subheader("Upload a cassava tuber image and choose a model to predict the PPD score.")

# Model selection
model_choice = st.radio("**Select Model**", list(models.keys()), horizontal=True)

# Upload image
uploaded_file = st.file_uploader("📤 Choose a cassava tuber image...", type=["jpg", "jpeg", "png"])

# Sidebar - Example images
st.sidebar.markdown("### 📁 Class Examples")
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

# Inference
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="🖼️ Uploaded Image", width=300)

    if model_choice == 'YOLOv8':
        try:
            temp_path = "temp.jpg"
            image.save(temp_path)
            results = models['YOLOv8'](temp_path)
            top1_class_index = results[0].probs.top1
            predicted_class = class_names[top1_class_index]
            st.success(f"✅ Predicted Class (YOLOv8): **{predicted_class.upper()}**")
        except Exception as e:
            st.error(f"❌ YOLOv8 Prediction failed: {e}")
    else:
        try:
            img_tensor = transform(image).unsqueeze(0)
            with torch.no_grad():
                output = models[model_choice](img_tensor)
                predicted_class = class_names[output.argmax().item()]
            st.success(f"✅ Predicted Class ({model_choice}): **{predicted_class.upper()}**")
        except Exception as e:
            st.error(f"❌ {model_choice} Prediction failed: {e}")
