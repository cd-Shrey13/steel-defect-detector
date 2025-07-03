import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
from fpdf import FPDF
import base64
import matplotlib.cm as cm
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
from datetime import datetime
import os


# Model & labels
MODEL_PATH = "model/steel_defect_classifier.h5"
model = tf.keras.models.load_model(MODEL_PATH)
class_names = ['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in-scale', 'scratches']

# Grad-CAM function
def make_gradcam_heatmap(img_array, model, last_conv_layer_name="Conv_1", pred_index=None):
    grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(last_conv_layer_name).output, model.output])
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

# PDF report

def generate_pdf(prediction, confidence, original_img_path, heatmap_img_path):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=14)
    
    # Title
    pdf.cell(200, 10, txt="Steel Surface Defect Detection Report", ln=True, align='C')
    pdf.ln(10)

    # Timestamp
    pdf.set_font("Arial", size=12)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    pdf.cell(200, 10, txt=f"Timestamp: {now}", ln=True)

    # Prediction and Confidence
    pdf.cell(200, 10, txt=f"Prediction: {prediction}", ln=True)
    pdf.cell(200, 10, txt=f"Confidence: {confidence:.2f}%", ln=True)
    pdf.ln(5)

    # Uploaded Image
    pdf.cell(200, 10, txt="Uploaded Image:", ln=True)
    pdf.image(original_img_path, x=10, w=90)
    pdf.ln(50)

    # Heatmap Image
    pdf.cell(200, 10, txt="Grad-CAM Heatmap:", ln=True)
    pdf.image(heatmap_img_path, x=10, w=90)

    # Save PDF
    pdf.output("report.pdf")

    # Read and return PDF as base64
    with open("report.pdf", "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")

    # Clean up temp files
    os.remove(original_img_path)
    os.remove(heatmap_img_path)
    os.remove("report.pdf")

    return encoded


# Title
st.set_page_config(page_title="Steel Defect Detector", layout="centered")
st.title("🔍 Surface Defect Detection - Hot Rolled Steel Strips")
st.caption("Upload an image or use your camera to detect steel defects using AI.")

# Image upload
uploaded_file = st.file_uploader("Upload Steel Surface Image", type=['jpg', 'jpeg', 'png'])

if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions)]
    confidence = np.max(predictions) * 100

    # Grad-CAM
    heatmap = make_gradcam_heatmap(img_array, model)
    heatmap_resized = cv2.resize(heatmap, (img.size[0], img.size[1]))
    heatmap_colored = np.uint8(255 * heatmap_resized)
    colored = cm.get_cmap("jet")(heatmap_colored)[:, :, :3]
    colored_heatmap = np.uint8(colored * 255)
    superimposed_img = cv2.addWeighted(np.array(img), 0.6, colored_heatmap, 0.4, 0)

    st.image(superimposed_img, caption="Grad-CAM Heatmap", use_column_width=True)

    st.markdown(f"### 🧠 Prediction: **{predicted_class.upper()}**")
    st.markdown(f"Confidence: `{confidence:.2f}%`")

    with st.expander("🔎 View all class probabilities"):
        for i, cls in enumerate(class_names):
            st.write(f"{cls}: {predictions[0][i]*100:.2f}%")

    # PDF Download
   # Save uploaded and heatmapped images
    original_img_path = "uploaded_image.jpg"
    heatmap_img_path = "gradcam_heatmap.jpg"
    image.save(original_img_path)
    Image.fromarray(superimposed_img).save(heatmap_img_path)

    # Generate PDF report and show download button
    pdf = generate_pdf(predicted_class, confidence, original_img_path, heatmap_img_path)

    st.download_button(
        label="📄 Download PDF Report",
        data=base64.b64decode(pdf),
        file_name="steel_defect_report.pdf",
        mime="application/pdf"
    )


# Live Camera Input
st.markdown("---")
st.header("🎥 Live Camera Mode (Real-Time Defect Detection)")

class LiveDefectDetector(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        resized = cv2.resize(img, (224, 224)) / 255.0
        input_tensor = np.expand_dims(resized, axis=0)
        preds = model.predict(input_tensor)
        label = class_names[np.argmax(preds)]
        conf = np.max(preds) * 100
        cv2.putText(img, f"{label.upper()} ({conf:.1f}%)", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        return img

webrtc_streamer(key="live", video_transformer_factory=LiveDefectDetector)
