import streamlit as st
from PIL import Image
import os
import torch
from torchvision import transforms

# Import your custom models
from model_resnet50 import ResNet50WithDropout
from model_efficientnet import EfficientNetV2SWithDropout
from ultralytics import YOLO

# Class names (ensure same order as used during training)
class_names = ['high', 'low', 'md', 'medium', 'zero']

# Preprocessing for torchvision models
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

@st.cache_resource
def load_models():
    models = {}

    # Load YOLOv8
    try:
        models['YOLOv8'] = YOLO("cassava_ppd_yolov8.pt")
    except Exception as e:
        st.warning(f"YOLOv8 load failed: {e}")

    # Load ResNet50
    try:
        resnet_model = ResNet50WithDropout(num_classes=len(class_names))
        resnet_model.load_state_dict(torch.load("resnetfinal_state_dict.pth", map_location="cpu"))
        resnet_model.eval()
        models['ResNet50'] = resnet_model
    except Exception as e:
        st.warning(f"ResNet50 load failed: {e}")

    # Load EfficientNet
    try:
        efficientnet_model = EfficientNetV2SWithDropout(num_classes=len(class_names))
        efficientnet_model.load_state_dict(torch.load("efficientnet_state_dict.pth", map_location="cpu"))
        efficientnet_model.eval()
        models['EfficientNetV2S'] = efficientnet_model
    except Exception as e:
        st.warning(f"EfficientNet load failed: {e}")

    return models

models = load_models()

# Streamlit UI
st.markdown("<h1 style='color:#198754;'>🧪 PPD Score Prediction from Tuber Images of Cassava</h1>", unsafe_allow_html=True)
st.subheader("Upload a cassava tuber image and choose a model to predict the PPD score.")

# Model selection
model_choice = st.selectbox("🔍 Select Model", list(models.keys()))

# File uploader
uploaded_file = st.file_uploader("📤 Choose a cassava tuber image...", type=["jpg", "jpeg", "png"])

# Example images in sidebar
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
            st.error(f"❌ Prediction failed: {e}")

    else:
        try:
            img_tensor = transform(image).unsqueeze(0)
            with torch.no_grad():
                output = models[model_choice](img_tensor)
                predicted_class = class_names[output.argmax().item()]
            st.success(f"✅ Predicted Class ({model_choice}): **{predicted_class.upper()}**")
        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")
