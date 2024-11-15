import os
import json
import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import base64

model_path = r"C:\Users\venga\Downloads\plant_disease_prediction_model.h5"
model = tf.keras.models.load_model(model_path)

class_indices = json.load(open('class_indices.json'))
disease_info = {
    "Apple___Apple_scab": {
        "cause": "Caused by the fungus Venturia inaequalis.",
        "treatment": "Use a fungicide containing captan or sulfur.",
        "growing_tips": "Ensure proper pruning for air circulation and avoid wet leaves.",
        "medicine_url": "https://www.amazon.com/s?k=captan+fungicide"
    },
    "Apple___Black_rot": {
        "cause": "Caused by the fungus Botryosphaeria obtusa.",
        "treatment": "Apply a fungicide like captan or copper-based fungicides.",
        "growing_tips": "Remove infected plant material and maintain dry foliage.",
        "medicine_url": "https://www.amazon.com/s?k=copper+fungicide"
    },
    "Apple___Cedar_apple_rust": {
        "cause": "Caused by the fungus Gymnosporangium juniperi-virginianae.",
        "treatment": "Use a fungicide like myclobutanil.",
        "growing_tips": "Remove nearby cedar trees to reduce spore spread.",
        "medicine_url": "https://www.amazon.com/s?k=myclobutanil+fungicide"
    },
    "Apple___healthy": {
        "cause": "No disease detected.",
        "treatment": "No medicine required.",
        "growing_tips": "Ensure proper spacing and pruning for air circulation.",
        "medicine_url": ""
    },
    "Blueberry___healthy": {
        "cause": "No disease detected.",
        "treatment": "No medicine required.",
        "growing_tips": "Ensure acidic soil and good drainage.",
        "medicine_url": ""
    },
    "Cherry_(including_sour)_Powdery_mildew": {
        "cause": "Caused by fungal pathogens that thrive in warm, dry conditions.",
        "treatment": "Apply fungicides like sulfur or neem oil.",
        "growing_tips": "Water plants at the base to keep leaves dry.",
        "medicine_url": "https://www.amazon.com/s?k=sulfur+fungicide"
    },
    "Cherry_(including_sour)_healthy": {
        "cause": "No disease detected.",
        "treatment": "No medicine required.",
        "growing_tips": "Ensure well-drained soil and full sunlight.",
        "medicine_url": ""
    },
    "Corn_(maize)_Cercospora_leaf_spot Gray_leaf_spot": {
        "cause": "Caused by the fungus Cercospora zeae-maydis.",
        "treatment": "Apply fungicides containing strobilurin or triazole.",
        "growing_tips": "Rotate crops and maintain proper plant spacing.",
        "medicine_url": "https://www.amazon.com/s?k=strobilurin+fungicide"
    },
    "Corn_(maize)Common_rust": {
        "cause": "Caused by the fungus Puccinia sorghi.",
        "treatment": "Use fungicides like mancozeb or propiconazole.",
        "growing_tips": "Plant rust-resistant varieties and rotate crops.",
        "medicine_url": "https://www.amazon.com/s?k=mancozeb+fungicide"
    },
    "Corn_(maize)_Northern_Leaf_Blight": {
        "cause": "Caused by the fungus Exserohilum turcicum.",
        "treatment": "Apply fungicides like mancozeb or chlorothalonil.",
        "growing_tips": "Avoid overhead irrigation to keep foliage dry.",
        "medicine_url": "https://www.amazon.com/s?k=chlorothalonil+fungicide"
    },
    "Corn_(maize)_healthy": {
        "cause": "No disease detected.",
        "treatment": "No medicine required.",
        "growing_tips": "Ensure adequate nitrogen levels and plant in full sun.",
        "medicine_url": ""
    },
    "Grape___Black_rot": {
        "cause": "Caused by the fungus Guignardia bidwellii.",
        "treatment": "Use fungicides containing myclobutanil or mancozeb.",
        "growing_tips": "Prune vines regularly for good air circulation.",
        "medicine_url": "https://www.amazon.com/s?k=myclobutanil+fungicide"
    },
    "Grape__Esca(Black_Measles)": {
        "cause": "Caused by a complex of fungal pathogens.",
        "treatment": "Apply fungicides containing myclobutanil.",
        "growing_tips": "Ensure proper drainage and avoid vine injury.",
        "medicine_url": "https://www.amazon.com/s?k=myclobutanil+fungicide"
    },
    "Grape__Leaf_blight(Isariopsis_Leaf_Spot)": {
        "cause": "Caused by the fungus Isariopsis clavispora.",
        "treatment": "Use fungicides like mancozeb or copper-based sprays.",
        "growing_tips": "Maintain plant health through balanced fertilization.",
        "medicine_url": "https://www.amazon.com/s?k=copper+fungicide"
    },
    "Grape___healthy": {
        "cause": "No disease detected.",
        "treatment": "No medicine required.",
        "growing_tips": "Plant in full sun and prune regularly.",
        "medicine_url": ""
    },
    "Orange__Haunglongbing(Citrus_greening)": {
        "cause": "Caused by a bacterial infection spread by the Asian citrus psyllid.",
        "treatment": "No cure, but managing the psyllid with insecticides can help.",
        "growing_tips": "Use pest control methods to manage psyllid populations.",
        "medicine_url": "https://www.amazon.com/s?k=insecticides"
    },
    "Peach___Bacterial_spot": {
        "cause": "Caused by the bacterium Xanthomonas campestris pv. pruni.",
        "treatment": "Use copper-based bactericides.",
        "growing_tips": "Ensure proper air circulation and remove infected leaves.",
        "medicine_url": "https://www.amazon.com/s?k=copper+bactericide"
    },
    "Peach___healthy": {
        "cause": "No disease detected.",
        "treatment": "No medicine required.",
        "growing_tips": "Ensure regular pruning and proper soil drainage.",
        "medicine_url": ""
    },
    "Pepper,bell__Bacterial_spot": {
        "cause": "Caused by Xanthomonas campestris pv. vesicatoria.",
        "treatment": "Use copper-based bactericides.",
        "growing_tips": "Space plants properly and avoid overhead watering.",
        "medicine_url": "https://www.amazon.com/s?k=copper+bactericide"
    },
    "Pepper,bell__healthy": {
        "cause": "No disease detected.",
        "treatment": "No medicine required.",
        "growing_tips": "Provide full sun and well-drained soil.",
        "medicine_url": ""
    },
    "Potato___Early_blight": {
        "cause": "Caused by the fungus Alternaria solani.",
        "treatment": "Use a fungicide like chlorothalonil.",
        "growing_tips": "Avoid wet leaves and rotate crops annually.",
        "medicine_url": "https://www.amazon.com/s?k=chlorothalonil+fungicide"
    },
    "Potato___Late_blight": {
        "cause": "Caused by the oomycete Phytophthora infestans.",
        "treatment": "Apply fungicides like mancozeb or chlorothalonil.",
        "growing_tips": "Ensure proper drainage and space plants well.",
        "medicine_url": "https://www.amazon.com/s?k=mancozeb+fungicide"
    },
    "Potato___healthy": {
        "cause": "No disease detected.",
        "treatment": "No medicine required.",
        "growing_tips": "Plant in well-drained soil and rotate crops.",
        "medicine_url": ""
    },
    "Raspberry___healthy": {
        "cause": "No disease detected.",
        "treatment": "No medicine required.",
        "growing_tips": "Provide good air circulation and prune regularly.",
        "medicine_url": ""
    },
    "Soybean___healthy": {
        "cause": "No disease detected.",
        "treatment": "No medicine required.",
        "growing_tips": "Ensure adequate nitrogen levels in the soil.",
        "medicine_url": ""
    },
    "Squash___Powdery_mildew": {
        "cause": "Caused by fungal pathogens that thrive in warm, dry conditions.",
        "treatment": "Apply fungicides like sulfur or neem oil.",
        "growing_tips": "Water plants at the base and avoid wet leaves.",
        "medicine_url": "https://www.amazon.com/s?k=sulfur+fungicide"
    },
    "Strawberry___Leaf_scorch": {
        "cause": "Caused by the fungus Diplocarpon earlianum.",
        "treatment": "Use fungicides containing myclobutanil or captan.",
        "growing_tips": "Ensure proper plant spacing and avoid wet leaves.",
        "medicine_url": "https://www.amazon.com/s?k=myclobutanil+fungicide"
    },
    "Strawberry___healthy": {
        "cause": "No disease detected.",
        "treatment": "No medicine required.",
        "growing_tips": "Provide well-drained soil and full sun.",
        "medicine_url": ""
    },
    "Tomato___Bacterial_spot": {
        "cause": "Caused by the bacterium Xanthomonas campestris pv. vesicatoria.",
        "treatment": "Use copper-based bactericides.",
        "growing_tips": "Rotate crops and avoid overhead watering.",
        "medicine_url": "https://www.amazon.com/s?k=copper+bactericide"
    },
    "Tomato___Early_blight": {
        "cause": "Caused by the fungus Alternaria solani.",
        "treatment": "Use a fungicide like chlorothalonil.",
        "growing_tips": "Avoid wet leaves and rotate crops.",
        "medicine_url": "https://www.amazon.com/s?k=chlorothalonil+fungicide"
    },
    "Tomato___Late_blight": {
        "cause": "Caused by the oomycete Phytophthora infestans.",
        "treatment": "Apply fungicides like mancozeb or chlorothalonil.",
        "growing_tips": "Ensure proper drainage and space plants well.",
        "medicine_url": "https://www.amazon.com/s?k=mancozeb+fungicide"
    },
    "Tomato___healthy": {
        "cause": "No disease detected.",
        "treatment": "No medicine required.",
        "growing_tips": "Plant in well-drained soil and provide full sun.",
        "medicine_url": ""
    }
}
def load_and_preprocess_image(image, target_size=(224, 224)):
    img = Image.open(image).resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32') / 255.
    return img_array

def predict_image_class(model, image, class_indices):
    preprocessed_img = load_and_preprocess_image(image)
    predictions = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_indices[str(predicted_class_index)]
    return predicted_class_name

def set_background(image_path):
    page_bg_img = f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{image_path}");
        background-size: cover;
    }}
    </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)

def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

bg_image_path = "C:/Users/venga/Downloads/—Pngtree—vibrant green tree leaves textured_13454816.jpg"
bg_image_base64 = get_base64_of_bin_file(bg_image_path)
set_background(bg_image_base64)

st.markdown("""
<div style='background-color: green; padding: 10px; border-radius: 5px;'>
    <h1 style='color: white; text-align: center;'>Plant Disease Detection</h1>
</div>
""", unsafe_allow_html=True)

capture_mode = st.radio("Choose Mode", ('Capture Photo', 'Upload Image'))
uploaded_file = None

if capture_mode == 'Capture Photo':
    st.markdown("<div style='margin: 10px; padding: 10px; border: 2px solid white; border-radius: 5px; background-color: rgba(255, 255, 255, 0.3);'>", unsafe_allow_html=True)
    captured_image = st.camera_input("Capture Photo")
    st.markdown("</div>", unsafe_allow_html=True)

    if captured_image is not None:
        st.image(captured_image, caption="Captured Image", use_column_width=True)
        if st.button('Scan Image'):
            st.markdown("<h2 style='color: red; font-weight: bold;'>This is not a leaf in my dataset.</h2>", unsafe_allow_html=True)

if capture_mode == 'Upload Image':
    st.markdown("<div style='margin: 10px; padding: 10px; border: 2px solid white; border-radius: 5px; background-color: rgba(255, 255, 255, 0.3);'>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Choose a plant leaf image...", type="jpg")
    st.markdown("</div>", unsafe_allow_html=True)

if uploaded_file is not None:
    prediction = predict_image_class(model, uploaded_file, class_indices)
    if prediction in disease_info:
        disease_data = disease_info[prediction]
        st.markdown(f"<h2 style='color: white; font-weight: bold;'>Predicted Disease: {prediction}</h2>", unsafe_allow_html=True)
        st.markdown(f"<div style='color: yellow; font-weight: bold;'><b>Cause:</b> {disease_data['cause']}</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='color: orange; font-weight: bold;'><b>Treatment:</b> {disease_data['treatment']}</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='color: blue; font-weight: bold;'><b>Growing Tips:</b> {disease_data['growing_tips']}</div>", unsafe_allow_html=True)

        if disease_data['medicine_url']:
            st.markdown(f"<a href='{disease_data['medicine_url']}' target='_blank'><button style='background-color: green; color: white; padding: 10px; border: none; border-radius: 5px;'>Buy Medicine</button></a>", unsafe_allow_html=True)
    else:
        st.markdown("<h2 style='color: red; font-weight: bold;'>This is not a recognized leaf image.</h2>", unsafe_allow_html=True)