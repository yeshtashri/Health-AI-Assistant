import streamlit as st
import pickle
import pandas as pd
import os
import numpy as np

# Set environment variables to manage CUDA memory
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Disable GPU
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

from fastai.vision.all import *
from PIL import Image
from rag import *
from model_fix import load_retinopathy_model_with_fallback, predict_retinopathy_robust

# Load Diabetes Prediction Model
diabetes_model_path = "PIMA/best_rf_model.pkl"
with open(diabetes_model_path, 'rb') as f:
    diabetes_model = pickle.load(f)

feature_columns_path = "PIMA/feature_columns.pkl"
with open(feature_columns_path, 'rb') as f:
    feature_columns = pickle.load(f)

# Load Diabetic Retinopathy Model with robust fallback
retinopathy_model, is_trained_model = load_retinopathy_model_with_fallback()
retinopathy_model_loaded = retinopathy_model is not None

# Function for Diabetes Prediction
def predict_diabetes(input_data):
    input_df = pd.DataFrame([input_data], columns=feature_columns)
    prediction = diabetes_model.predict(input_df)
    return prediction[0]

# Function for Retinopathy Prediction using robust approach
def predict_retinopathy(image_path):
    return predict_retinopathy_robust(retinopathy_model, image_path, is_trained_model)

diabetes_vector_db_path = "PIMA/vector_db"

# Diabetic Retinopathy Dataset
retinopathy_vector_db_path = "Two classes/vector_db"

diabetes_vector_db_path = "PIMA/vector_db"
retinopathy_vector_db_path = "Two classes/vector_db"

diabetes_vector_db = load_vector_store(diabetes_vector_db_path)
retinopathy_vector_db = load_vector_store(retinopathy_vector_db_path)

# Inject Three.js and GSAP animated background and parallax effect
st.markdown('''
<style>
#canvas-container {
    position: fixed;
    top: 0; left: 0; width: 100vw; height: 100vh; z-index: -1;
}

/* App background color */
.stApp, .main .block-container {
    background: #f5f5f5 !important;
}

/* Make all input/search/select boxes white */
input, textarea, .stTextInput > div > div > input, .stSelectbox > div > div > div, .stTextArea > div > textarea, .stNumberInput > div > input, .stDateInput > div > input, .stFileUploader > div, .stMultiSelect > div > div > div {
    background: white !important;
    color: black !important;
    border-radius: 8px !important;
    border: 1px solid #ccc !important;
}

/* Ensure selectbox dropdown is white */
.stSelectbox [data-baseweb="select"] > div {
    background: white !important;
    color: black !important;
}

/* Custom background colors and text */
.main .block-container {
    background: linear-gradient(135deg, #e3f2fd 0%, #f5f5f5 100%);
    color: black;
}

/* Override Streamlit default colors */
.stApp {
    background: linear-gradient(135deg, #e3f2fd 0%, #f5f5f5 100%);
}

/* Make headings dark blue */
h1, h2, h3 {
    color: #0d47a1 !important;
}

/* Make other text black */
p, label, div {
    color: black !important;
}

/* Style sidebar */
.sidebar .sidebar-content {
    background: linear-gradient(135deg, #bbdefb 0%, #e0e0e0 100%);
    color: black;
}

/* Style buttons */
.stButton > button {
    background: linear-gradient(45deg, #2196f3, #64b5f6);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 10px 20px;
    font-weight: bold;
}

.stButton > button:hover {
    background: linear-gradient(45deg, #1976d2, #42a5f5);
    transform: translateY(-2px);
    box-shadow: 0 4px 8px rgba(0,0,0,0.2);
}

/* Style input fields */
.stTextInput > div > div > input {
    background: rgba(255, 255, 255, 0.8);
    border: 1px solid #ccc;
    color: black;
}

/* Style number inputs */
.stNumberInput > div > div > input {
    background: rgba(255, 255, 255, 0.8);
    border: 1px solid #ccc;
    color: black;
}

/* Style file uploader */
.stFileUploader > div {
    background: rgba(255, 255, 255, 0.8);
    border: 1px solid #ccc;
    color: black;
}

/* Style selectbox */
.stSelectbox > div > div > div {
    background: white !important;
    color: black !important;
}

/* Style navigation bar */
.sidebar .sidebar-content {
    background: white !important;
    color: black !important;
}

/* Style sidebar elements */
.sidebar .sidebar-content .stSelectbox > div > div > div {
    background: white !important;
    color: black !important;
}

.sidebar .sidebar-content label {
    color: black !important;
}

.sidebar .sidebar-content .stMarkdown {
    color: black !important;
}

/* Override any sidebar text colors */
.sidebar .sidebar-content * {
    color: black !important;
}

/* Additional navigation bar styling */
.sidebar .sidebar-content .stSelectbox > div > div > div > div {
    background: white !important;
    color: black !important;
}

.sidebar .sidebar-content .stSelectbox > div > div > div > div > div {
    background: white !important;
    color: black !important;
}

/* Force white background for all sidebar containers */
.sidebar .sidebar-content > div {
    background: white !important;
}

/* Ensure selectbox text is black */
.sidebar .sidebar-content .stSelectbox label {
    color: black !important;
    font-weight: bold;
}

/* Sidebar (menu/navigation bar) background and text color */
.stSidebar, .sidebar .sidebar-content {
    background: white !important;
    color: black !important;
}

/* Sidebar elements */
.stSidebar * {
    color: black !important;
}

/* Sidebar selectbox, input, etc. */
.stSidebar input, .stSidebar textarea, .stSidebar select, .stSidebar .stSelectbox > div > div > div {
    background: white !important;
    color: black !important;
    border-radius: 8px !important;
    border: 1px solid #ccc !important;
}

.gradient-heading {
    background: linear-gradient(90deg, #64b5f6 0%, #0d47a1 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    text-fill-color: transparent;
}

/* Dropdown menu (selectbox) styling */
.stSelectbox > div > div > div, .stSelectbox [data-baseweb="select"] > div {
    background: white !important;
    color: black !important;
    border-radius: 8px !important;
    border: 1px solid #ccc !important;
}
.stSelectbox [data-baseweb="select"] [role="listbox"] {
    background: white !important;
    color: black !important;
}

/* Animated gradient background for Home page */
.home-bg-animated {
    position: fixed;
    top: 0; left: 0; width: 100vw; height: 100vh;
    z-index: -2;
    background: linear-gradient(270deg, #e3f2fd, #bbdefb, #64b5f6, #1976d2, #0d47a1, #e3f2fd);
    background-size: 1200% 1200%;
    animation: gradientMove 18s ease-in-out infinite;
}
@keyframes gradientMove {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}
/* Optional: floating circles */
.circle {
    position: absolute;
    border-radius: 50%;
    opacity: 0.18;
    animation: float 12s infinite ease-in-out;
}
.circle1 { width: 180px; height: 180px; left: 10vw; top: 20vh; background: #1976d2; animation-delay: 0s; }
.circle2 { width: 120px; height: 120px; left: 70vw; top: 60vh; background: #64b5f6; animation-delay: 3s; }
.circle3 { width: 90px; height: 90px; left: 40vw; top: 80vh; background: #0d47a1; animation-delay: 6s; }
@keyframes float {
    0% { transform: translateY(0px) scale(1); }
    50% { transform: translateY(-40px) scale(1.1); }
    100% { transform: translateY(0px) scale(1); }
}
/* Glassmorphism card effect */
.glass-card {
    background: rgba(255,255,255,0.25);
    box-shadow: 0 8px 32px 0 rgba(31,38,135,0.18);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border-radius: 18px;
    border: 1px solid rgba(255,255,255,0.28);
    padding: 2.5rem 2.5rem 2rem 2.5rem;
    margin: 2rem auto 1.5rem auto;
    max-width: 600px;
    animation: fadeInUp 1.2s cubic-bezier(.39,.575,.56,1) both;
}
@keyframes fadeInUp {
    0% { opacity: 0; transform: translateY(40px); }
    100% { opacity: 1; transform: translateY(0); }
}
.index-list {
    margin-top: 1.5rem;
    padding-left: 0.5rem;
}
.index-list a {
    color: #1976d2;
    font-weight: 600;
    text-decoration: none;
    font-size: 1.1rem;
    transition: color 0.2s;
}
.index-list a:hover {
    color: #0d47a1;
    text-decoration: underline;
}

/* File uploader and dark backgrounds to white, text to black */
.stFileUploader, .stFileUploader > div, .stFileUploader [data-testid="stFileDropzone"] {
    background: white !important;
    color: black !important;
    border-radius: 8px !important;
    border: 1px solid #ccc !important;
}
.stFileUploader [data-testid="stFileDropzone"] * {
    color: black !important;
}

/* Selectbox dropdown menu to white, text to black */
.stSelectbox > div > div > div, .stSelectbox [data-baseweb="select"] > div, .stSelectbox [data-baseweb="select"] [role="listbox"] {
    background: white !important;
    color: black !important;
}

/* General override for any dark backgrounds */
[data-testid="stSidebar"], .stApp, .main .block-container, .stButton, .stTextInput, .stTextArea, .stNumberInput, .stSelectbox, .stMultiSelect, .stDateInput, .stFileUploader {
    background: white !important;
    color: black !important;
}

/* Ensure all text is black */
h1, h2, h3, h4, h5, h6, p, label, div, span, li, ul, ol, input, textarea, select, .stMarkdown, .stText, .stButton, .stFileUploader, .stSelectbox {
    color: black !important;
}
</style>
<div id="canvas-container"></div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/gsap/3.12.2/gsap.min.js"></script>
<script>
// Three.js Background
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
const renderer = new THREE.WebGLRenderer({ alpha: true });
renderer.setSize(window.innerWidth, window.innerHeight);
document.getElementById('canvas-container').appendChild(renderer.domElement);

// Create particles
const particlesGeometry = new THREE.BufferGeometry();
const particlesCount = 5000;
const posArray = new Float32Array(particlesCount * 3);
for(let i = 0; i < particlesCount * 3; i++) {
    posArray[i] = (Math.random() - 0.5) * 5;
}
particlesGeometry.setAttribute('position', new THREE.BufferAttribute(posArray, 3));
const particlesMaterial = new THREE.PointsMaterial({
    size: 0.005,
    color: '#4ecdc4'
});
const particlesMesh = new THREE.Points(particlesGeometry, particlesMaterial);
scene.add(particlesMesh);
camera.position.z = 2;


</script>
''', unsafe_allow_html=True)

# Streamlit App
st.markdown('<h1 class="gradient-heading">Diabetes and Retinopathy Detection & Consultation</h1>', unsafe_allow_html=True)

# Sidebar for user selection
page = st.sidebar.selectbox(
    "Choose an option",
    ("Home", "Diabetes Prediction", "Retinopathy Detection", "Consultation")
)

if page == "Home":
    st.markdown('''
    <style>
    /* Animated gradient background for Home page */
    .home-bg-animated {
        position: fixed;
        top: 0; left: 0; width: 100vw; height: 100vh;
        z-index: -2;
        background: linear-gradient(270deg, #e3f2fd, #bbdefb, #64b5f6, #1976d2, #0d47a1, #e3f2fd);
        background-size: 1200% 1200%;
        animation: gradientMove 18s ease-in-out infinite;
    }
    @keyframes gradientMove {
        0% {background-position: 0% 50%;}
        50% {background-position: 100% 50%;}
        100% {background-position: 0% 50%;}
    }
    .circle {
        position: absolute;
        border-radius: 50%;
        opacity: 0.18;
        animation: float 12s infinite ease-in-out;
    }
    .circle1 { width: 180px; height: 180px; left: 10vw; top: 20vh; background: #1976d2; animation-delay: 0s; }
    .circle2 { width: 120px; height: 120px; left: 70vw; top: 60vh; background: #64b5f6; animation-delay: 3s; }
    .circle3 { width: 90px; height: 90px; left: 40vw; top: 80vh; background: #0d47a1; animation-delay: 6s; }
    @keyframes float {
        0% { transform: translateY(0px) scale(1); }
        50% { transform: translateY(-40px) scale(1.1); }
        100% { transform: translateY(0px) scale(1); }
    }
    /* Glassmorphism card effect */
    .glass-card {
        background: rgba(255,255,255,0.25);
        box-shadow: 0 8px 32px 0 rgba(31,38,135,0.18);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border-radius: 18px;
        border: 1px solid rgba(255,255,255,0.28);
        padding: 2.5rem 2.5rem 2rem 2.5rem;
        margin: 2rem auto 1.5rem auto;
        max-width: 600px;
        animation: fadeInUp 1.2s cubic-bezier(.39,.575,.56,1) both;
    }
    @keyframes fadeInUp {
        0% { opacity: 0; transform: translateY(40px); }
        100% { opacity: 1; transform: translateY(0); }
    }
    .index-list {
        margin-top: 1.5rem;
        padding-left: 0.5rem;
    }
    .index-list a {
        color: #1976d2;
        font-weight: 600;
        text-decoration: none;
        font-size: 1.1rem;
        transition: color 0.2s;
    }
    .index-list a:hover {
        color: #0d47a1;
        text-decoration: underline;
    }
    </style>
    <div class="home-bg-animated"></div>
    <div class="circle circle1"></div>
    <div class="circle circle2"></div>
    <div class="circle circle3"></div>
    <div class="glass-card">
        <h1 class="gradient-heading" style="margin-bottom: 0.5rem;">Diabetes and Retinopathy Detection & Consultation</h1>
        <p style="font-size:1.15rem; margin-bottom:1.5rem;">Welcome to your all-in-one AI-powered health assistant. This app helps you predict diabetes, detect diabetic retinopathy from retina images, and get instant clinical consultation using advanced AI.</p>
        <h2 style="color:#1976d2; font-size:1.3rem; margin-bottom:0.7rem;">Index</h2>
        <ul class="index-list">
            <li><a href="#diabetes-prediction">Diabetes Prediction</a></li>
            <li><a href="#retinopathy-detection">Retinopathy Detection</a></li>
            <li><a href="#consultation">Consultation (Q&A)</a></li>
        </ul>
    </div>
    <div class="glass-card" style="max-width: 600px;">
        <h3 style="color:#0d47a1; margin-bottom:0.5rem;">How to Use</h3>
        <ol style="font-size:1.05rem;">
            <li>Select a feature from the sidebar menu or the index above.</li>
            <li>Follow the instructions on each page.</li>
            <li>Enjoy a modern, easy-to-use interface with beautiful visuals and clear results.</li>
        </ol>
        <hr style="margin:1.2rem 0; border:0; border-top:1px solid #bbdefb;">
        <p style="font-size:1rem; color:#333;">This project leverages machine learning, deep learning, and retrieval-augmented generation to provide clinical insights and support.</p>
    </div>
    ''', unsafe_allow_html=True)

elif page == "Diabetes Prediction":
    st.header("Diabetes Prediction")
    user_input = {}
    for feature in feature_columns:
        user_input[feature] = st.number_input(f"Enter {feature}", value=0.0)
    
    if st.button("Predict Diabetes"):
        prediction = predict_diabetes(user_input)
        st.write(f"Predicted Diabetes Risk: {'Diabetic' if prediction == 1 else 'Non-Diabetic'}")

elif page == "Retinopathy Detection":
    st.header("Diabetic Retinopathy Detection")
    
    if not retinopathy_model_loaded:
        st.error("❌ Retinopathy detection is currently unavailable.")
        st.info("🔧 **Possible Solutions:**")
        st.info("1. The model needs to be retrained on a Linux system")
        st.info("2. The model file needs to be converted to be cross-platform compatible")
        st.info("3. Use a different model format (e.g., ONNX, TorchScript)")
        
        # Provide a demo option
        if st.button("Try Demo with Sample Image"):
            st.info("Demo mode: This would normally analyze a retina image for diabetic retinopathy.")
            st.info("Features detected: Blood vessels, optic disc, macula")
            st.info("Sample result: No diabetic retinopathy detected")
    else:
        uploaded_file = st.file_uploader("Upload a Retina Image", type=["png", "jpg", "jpeg"])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Retina Image')
            
            if st.button("Analyze Image"):
                with st.spinner("Analyzing image..."):
                    image_path = f"temp_image.{uploaded_file.name.split('.')[-1]}"
                    image.save(image_path)
                    pred_class, probs = predict_retinopathy(image_path)
                    
                    if pred_class != "Model not available" and pred_class != "Prediction failed":
                        st.success(f"✅ Analysis Complete!")
                        st.write(f"**Predicted Class:** {pred_class}")
                        
                        # Add interpretation
                        if pred_class == "DR":
                            st.warning("⚠️ **Diabetic Retinopathy Detected**")
                            st.info("Please consult with an ophthalmologist for proper diagnosis and treatment.")
                        else:
                            st.success("✅ **No Diabetic Retinopathy Detected**")
                            st.info("Continue regular eye checkups as recommended by your doctor.")
                    else:
                        st.error(f"❌ {pred_class}")
                    
                    # Clean up temporary file
                    try:
                        os.remove(image_path)
                    except:
                        pass

elif page == "Consultation":
    st.header("Clinical Consultation")
    user_query = st.text_input("Ask a question about diabetes or retinopathy:")

    if st.button("Get Answer"):
        if user_query:
            response = extract_last_answer(rag_query(diabetes_vector_db, user_query))
            st.write(response)
        else:
            st.write("Please enter a query.")