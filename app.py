import streamlit as st
import numpy as np
from ultralytics import YOLO
from PIL import Image, ImageDraw

# Custom CSS for Pakistan Army Uniform Colors
st.markdown("""
<style>
    body {
        background: linear-gradient(to right, #4B5320, #2E3B16, #1A260E); /* Army Green Shades */
        color: #ffffff;
        font-family: Arial, sans-serif;
    }
    .main { 
        background: #ffffff; 
        color: #000000; 
        padding: 20px;
    }
    h1 {
        color: #ffffff;
        text-align: center;
    }
    .upload-btn {
        background-color: #4B5320 !important;
        color: white !important;
    }
    .sidebar .sidebar-content {
        background: #2E3B16 !important;
        color: #ffffff;
    }
    .footer {
        background-color: #1A260E;
        color: #ffffff;
        padding: 20px;
        text-align: center;
    }
    .bold-text {
        font-weight: bold;
        font-size: 18px;
    }
</style>
""", unsafe_allow_html=True)

# ✅ Load YOLO Model
@st.cache_resource
def load_model():
    model_path = "E:\PROJECT\military_assets_object dection project\yolo11n (5).pt"
    return YOLO(model_path)

model = load_model()

# Define Class Labels & Colors
class_labels = ['camouflage_soldier', 'weapon', 'military_tank', 'military_truck', 'military_vehicle', 'civilian', 'soldier', 'civilian_vehicle', 'military_artillery', 'trench', 'military_aircraft', 'military_warship']
class_colors = ['#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF', '#00FFFF', '#800000', '#808000', '#008000', '#800080', '#008080', '#000080']
  # Black-Golden Gradient
# Main App
st.title("🪖 Military Assets Detection AI")
st.subheader("Upload an image to detect military and civilian assets")

# Sidebar
with st.sidebar:
    st.header("Settings")
    confidence = st.slider("Confidence Threshold", 0.1, 1.0, 0.5)

# Image Upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"], key="uploader")

if uploaded_file is not None:
    # Read Image
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)
    
    # Run Detection ✅
    results = model.predict(img_array)
    
    # Draw bounding boxes
    draw = ImageDraw.Draw(image)
    detected_objects = []
    
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get bounding box
            cls = int(box.cls[0])  # Class ID
            label = class_labels[cls] if cls < len(class_labels) else "Unknown"
            color = class_colors[cls] if cls < len(class_colors) else "#000000"
            detected_objects.append(f"<span style='color:{color}; font-weight:bold;'>{label}</span>")
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            draw.text((x1, y1 - 10), label, fill=color)

    # Display Results ✅
    st.image(image, caption="**Detection Results**", use_column_width=True)

    # Show Detection Details
    st.subheader("Detection Statistics")
    num_objects = len(detected_objects)
    st.write(f"**Detected Objects: {num_objects}**")
    st.markdown(f"Objects Detected: {', '.join(detected_objects)}", unsafe_allow_html=True)

# Footer with "Data Scientist"
st.markdown("""
<div class="footer">
    <p>Made with ❤️ by MOHID_KHAN, Data Scientist</p>
    <p>Email: Mohidadil24@gmail.com | Github: <a href="https://github.com/mohidadil" target="_blank">mohidadil</a></p>
</div>
""", unsafe_allow_html=True)
