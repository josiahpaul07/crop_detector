import streamlit as st
import requests
import google.generativeai as genai

# App title
st.title("🌾 Crop Disease Detection & Cure Recommender")

# --- Configuration ---
# Custom Vision API settings
CUSTOM_VISION_ENDPOINT = "https://eastus.api.cognitive.microsoft.com/customvision/v3.0/Prediction/e507e178-1390-468a-9c8c-81f228dcadcc/classify/iterations/crop_diseases_detection/image"
PREDICTION_KEY = "c43480edd8d047ab8546b37fea8e89c9"

# Gemini API settings
GEMINI_API_KEY = "AIzaSyDWN-lXdhrNSD4arKrFA6d581eKKz0iK8c"
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("ggemini-2.0-flash")

# Headers for Custom Vision
HEADERS = {
    "Content-Type": "application/octet-stream",
    "Prediction-Key": PREDICTION_KEY
}

# --- Functions ---
def get_disease_prediction(image_bytes):
    """Send image to Custom Vision API."""
    response = requests.post(CUSTOM_VISION_ENDPOINT, headers=HEADERS, data=image_bytes)
    if response.status_code != 200:
        st.error(f"Custom Vision Error: {response.text}")
        return None
    predictions = response.json().get("predictions", [])
    if not predictions:
        return None
    return max(predictions, key=lambda x: x['probability'])

def get_cure_recommendation(disease_name):
    """Get treatment recommendations using Gemini LLM."""
    prompt = f"The plant has been diagnosed with {disease_name}. Provide a recommended cure, including organic and chemical treatment options."
    response = gemini_model.generate_content(prompt)
    return response.text if response else "No recommendation available."

# --- Streamlit Interface ---
uploaded_file = st.file_uploader("Upload a crop image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    image_bytes = uploaded_file.read()
    
    with st.spinner('🔍 Detecting disease...'):
        top_prediction = get_disease_prediction(image_bytes)
    
    if top_prediction:
        disease_name = top_prediction.get("tagName", "Unknown Disease")
        probability = round(top_prediction.get("probability", 0) * 100, 2)
        
        st.success(f"🌿 **Detected Disease:** {disease_name} ({probability}%)")
        
        with st.spinner('💡 Generating cure recommendation...'):
            cure_recommendation = get_cure_recommendation(disease_name)
        
        st.subheader("🩺 Recommended Cure")
        st.write(cure_recommendation)
    else:
        st.error("No predictions found or an error occurred.")
