import streamlit as st
import numpy as np
import cv
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input

# =========================
# LOAD MODEL
# =========================
# model = load_model(r"C:\Users\haris\Desktop\aug_healthy_test\mobilenet_balancetd_improved.keras")
import os
import gdown
from tensorflow.keras.models import load_model

MODEL_PATH = "model.keras"

# Download model from Google Drive (only first time)
if not os.path.exists(MODEL_PATH):
    url = "https://drive.google.com/uc?id=1Od4FGU85tJFP-N50CsG5UqZlIQSrAS2Z"
    gdown.download(url, MODEL_PATH, quiet=False)

# Load model
@st.cache_resource
def load_my_model():
    return load_model(MODEL_PATH)

model = load_my_model()
# =========================
# PAGE SETTINGS
# =========================
st.set_page_config(page_title="Plant Stress Detection", layout="centered")

st.markdown(
    "<h1 style='text-align: center; color: green;'>🌿 Plant Stress Detection System</h1>",
    unsafe_allow_html=True
)

st.write("Upload a leaf image to detect plant stress and visualize affected regions.")

# =========================
# FIXED GRAD-CAM FUNCTION
# =========================
def get_gradcam(img_array, model):

    # 🔥 Find last convolution layer automatically
    last_conv_layer = None
    for layer in reversed(model.layers):
        if "conv" in layer.name:
            last_conv_layer = layer.name
            break

    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        class_idx = tf.argmax(predictions[0])
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)

    # ✅ Safe now (conv layer → 4D tensor)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap) + 1e-8

    if isinstance(heatmap, tf.Tensor):
       heatmap = heatmap.numpy()
    return heatmap
# =========================
# FILE UPLOAD
# =========================
uploaded_file = st.file_uploader("📤 Upload Leaf Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:

    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    st.image(img, caption="📷 Uploaded Image", width=300)

    # =========================
    # PREPROCESS
    # =========================
    img_resized = cv2.resize(img, (288, 288))  # ✅ correct size
    img_input = preprocess_input(img_resized)
    img_input = np.expand_dims(img_input, axis=0)

    # =========================
    # PREDICTION (MULTI-CLASS)
    # =========================
    pred = model.predict(img_input)

    class_index = np.argmax(pred)

    classes = ["Healthy", "Nutrient Stress", "Pathogenic Stress"]
    label = classes[class_index]

    st.write("🔍 Prediction Probabilities:", pred)

    # =========================
    # RESULT DISPLAY
    # =========================
    if label == "Healthy":
        st.success("🌿 Healthy Plant")

    elif label == "Nutrient Stress":
        st.warning("🟡 Nutrient Stress Detected")

    else:
        st.error("🔴 Pathogenic Stress Detected")

    # =========================
    # GRAD-CAM VISUALIZATION
    # =========================
    heatmap = get_gradcam(img_input, model)

    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)

    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    superimposed_img = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

    st.image(superimposed_img, caption="🔥 Grad-CAM Visualization", width=300)

    # =========================
    # RECOMMENDATION
    # =========================
    st.subheader("💡 Recommendation")

    if label == "Healthy":
        st.info("No action needed. Plant is healthy.")

    elif label == "Nutrient Stress":
        st.write("Apply fertilizers and improve soil nutrients.")

    else:
        st.write("Apply pesticide or treat plant disease.")
