import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import gdown
file_id="1aaFrnBq5jW_AQoInHWfB1OG6WcfpN60b"
url="https://drive.google.com/file/d/1aaFrnBq5jW_AQoInHWfB1OG6WcfpN60b/view?usp=sharing"

# Define model path
MODEL_PATH = "trained_plant_disease_model.keras"
model=tf.keras.models.load_model(model_path)

if not os.path.exists(model_path):
    st.warning("Downloading model from Google Drive...")
    gdown.download(url, model_path,quiet=False)
                   

def model_prediction(test_image):
    model = tf.keras.models.load_model(MODEL_PATH)
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = img_to_array(image)
    input_arr = np.expand_dims(input_arr, axis=0)  # Convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions)  # Return index of max element

# Custom CSS for a light green effect
st.markdown(
    """
    <style>
    body, .stApp {
        background: linear-gradient(135deg, #b7e4c7, #74c69d) !important; /* Light green gradient */
    }
    .main {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0px 4px 12px rgba(0, 120, 0, 0.2);
        margin: auto;
        max-width: 700px;
    }
    h1 {
        color: #40916c; /* Softer green */
        text-align: center;
        text-shadow: 1px 1px 3px rgba(0, 100, 0, 0.2);
    }
    .stButton button {
        background: linear-gradient(135deg, #a8dadc, #52b788) !important;
        color: white !important;
        font-size: 18px !important;
        border-radius: 10px !important;
        box-shadow: 0px 3px 6px rgba(0, 120, 0, 0.3);
    }
    .stImage img {
        border-radius: 10px;
        box-shadow: 0px 4px 10px rgba(0, 80, 0, 0.3);
    }
    .stSuccess {
        background: linear-gradient(135deg, #d8f3dc, #95d5b2) !important;
        color: #1b4332 !important;
        padding: 12px;
        border-radius: 12px;
        box-shadow: 0px 3px 6px rgba(0, 100, 0, 0.3);
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Load the trained model
try:
    model = tf.keras.models.load_model(MODEL_PATH)
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

# Define class labels
class_labels = ["Healthy", "Early Blight", "Late Blight"]

def preprocess_image(img):
    img = img.resize((128, 128))  # Resize to match model input
    img = img_to_array(img)  # Convert image to array
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Streamlit UI
st.markdown('<div class="main">', unsafe_allow_html=True)
st.title("🥔 Potato Leaf 🌿 Disease Detection")
st.write("UPLOAD AN IMAGE OF A POTATO LEAF 🌱 TO CHECK FOR DISEASE.")

# Upload image
uploaded_file = st.file_uploader("📂 Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="📷 Uploaded Image", use_container_width=True)

    # Button to trigger prediction
    if st.button("🔍 Predict"):
        processed_img = preprocess_image(img)
        prediction = model.predict(processed_img)
        predicted_class = class_labels[np.argmax(prediction)]
        confidence_scores = {class_labels[i]: round(float(prediction[0][i]) * 100, 2) for i in range(len(class_labels))}

        # Display result
        st.markdown(f'<p class="stSuccess">🏷 Prediction: <b>{predicted_class}</b></p>', unsafe_allow_html=True)

        # Show confidence scores
        st.write("🔢 *Confidence Scores:*")
        for label, score in confidence_scores.items():
            st.markdown(f"<p style='color:#1b4332; font-size:16px;'>🔹 <b>{label}:</b> {score}%</p>", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
