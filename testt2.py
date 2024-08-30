import streamlit as st
import pickle
import numpy as np
import os

symptoms=np.array(['itching', 'skin_rash', 'nodal_skin_eruptions',
       'continuous_sneezing', 'shivering', 'chills', 'joint_pain',
       'stomach_pain', 'acidity', 'ulcers_on_tongue', 'muscle_wasting',
       'vomiting', 'burning_micturition', 'spotting_ urination',
       'fatigue', 'weight_gain', 'anxiety', 'cold_hands_and_feets',
       'mood_swings', 'weight_loss', 'restlessness', 'lethargy',
       'patches_in_throat', 'irregular_sugar_level', 'cough',
       'high_fever', 'sunken_eyes', 'breathlessness', 'sweating',
       'dehydration', 'indigestion', 'headache', 'yellowish_skin',
       'dark_urine', 'nausea', 'loss_of_appetite', 'pain_behind_the_eyes',
       'back_pain', 'constipation', 'abdominal_pain', 'diarrhoea',
       'mild_fever', 'yellow_urine', 'yellowing_of_eyes',
       'acute_liver_failure', 'fluid_overload', 'swelling_of_stomach',
       'swelled_lymph_nodes', 'malaise', 'blurred_and_distorted_vision',
       'phlegm', 'throat_irritation', 'redness_of_eyes', 'sinus_pressure',
       'runny_nose', 'congestion', 'chest_pain', 'weakness_in_limbs',
       'fast_heart_rate', 'pain_during_bowel_movements',
       'pain_in_anal_region', 'bloody_stool', 'irritation_in_anus',
       'neck_pain', 'dizziness', 'cramps', 'bruising', 'obesity',
       'swollen_legs', 'swollen_blood_vessels', 'puffy_face_and_eyes',
       'enlarged_thyroid', 'brittle_nails', 'swollen_extremeties',
       'excessive_hunger', 'extra_marital_contacts',
       'drying_and_tingling_lips', 'slurred_speech', 'knee_pain',
       'hip_joint_pain', 'muscle_weakness', 'stiff_neck',
       'swelling_joints', 'movement_stiffness', 'spinning_movements',
       'loss_of_balance', 'unsteadiness', 'weakness_of_one_body_side',
       'loss_of_smell', 'bladder_discomfort', 'foul_smell_of urine',
       'continuous_feel_of_urine', 'passage_of_gases', 'internal_itching',
       'toxic_look_(typhos)', 'depression', 'irritability', 'muscle_pain',
       'altered_sensorium', 'red_spots_over_body', 'belly_pain',
       'abnormal_menstruation', 'dischromic _patches',
       'watering_from_eyes', 'increased_appetite', 'polyuria',
       'family_history', 'mucoid_sputum', 'rusty_sputum',
       'lack_of_concentration', 'visual_disturbances',
       'receiving_blood_transfusion', 'receiving_unsterile_injections',
       'coma', 'stomach_bleeding', 'distention_of_abdomen',
       'history_of_alcohol_consumption', 'fluid_overload.1',
       'blood_in_sputum', 'prominent_veins_on_calf', 'palpitations',
       'painful_walking', 'pus_filled_pimples', 'blackheads', 'scurring',
       'skin_peeling', 'silver_like_dusting', 'small_dents_in_nails',
       'inflammatory_nails', 'blister', 'red_sore_around_nose',
       'yellow_crust_ooze'])

diseases=['(vertigo) Paroymsal  Positional Vertigo' ,'AIDS', 'Acne',
 'Alcoholic hepatitis' ,'Allergy' ,'Arthritis', 'Bronchial Asthma',
 'Cervical spondylosis' ,'Chicken pox', 'Chronic cholestasis' ,'Common Cold',
 'Dengue', 'Diabetes ' ,'Dimorphic hemmorhoids(piles)' ,'Drug Reaction',
 'Fungal infection', 'GERD', 'Gastroenteritis', 'Heart attack', 'Hepatitis B',
 'Hepatitis C', 'Hepatitis D', 'Hepatitis E', 'Hypertension ',
 'Hyperthyroidism', 'Hypoglycemia', 'Hypothyroidism', 'Impetigo' ,'Jaundice',
 'Malaria', 'Migraine', 'Osteoarthristis', 'Paralysis (brain hemorrhage)',
 'Peptic ulcer diseae', 'Pneumonia' ,'Psoriasis', 'Tuberculosis', 'Typhoid',
 'Urinary tract infection', 'Varicose veins', 'hepatitis A']

# Data Dictionary
symptom_index = { " ".join([i.capitalize() for i in value.split("_")]): index for index, value in enumerate(symptoms)}
data_dict = {
    "symptom_index": symptom_index,
    "predictions_classes": diseases
}

# Function to Predict Disease
def predictDisease(symptoms):
    model_path = r"final_rf_model.pkl"
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    symptoms = symptoms.split(",")
    input_data = [0] * len(data_dict["symptom_index"])
    
    for symptom in symptoms:
        symptom = symptom.strip()
        if symptom in data_dict["symptom_index"]:
            index = data_dict["symptom_index"][symptom]
            input_data[index] = 1
    
    input_data = np.array(input_data).reshape(1, -1)
    rf_prediction = data_dict["predictions_classes"][model.predict(input_data)[0]]
    
    return rf_prediction

# Streamlit Page Layout
st.set_page_config(layout="wide")

# Loading CSS and JS files
css_path = 'home.css'
js_path = 'home.js'

if os.path.exists(css_path):
    with open(css_path) as c:
        st.markdown(f"<style>{c.read()}</style>", unsafe_allow_html=True)
else:
    st.warning("CSS file not found!")

if os.path.exists(js_path):
    with open(js_path) as j:
        st.markdown(f"<script>{j.read()}</script>", unsafe_allow_html=True)
else:
    st.warning("JS file not found!")

# HTML Content
html_content = """
<header>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Roboto:wght@400;500&family=Tajawal:wght@300&display=swap" rel="stylesheet">
    <div class="container">
        <div class="logo">
            <h2>MedDetect</h2>
        </div>
        <nav class="nav-a">
            <ul>
                <li><a href="home.html">Home</a></li>
                <li><a href="#about">About</a></li>
                <li><a href="#contact">Contact</a></li>
            </ul>
        </nav>
    </div>
</header>
<section class="slider" id="slider">
    <div class="slider-text">Welcome to the disease prediction platform where proactive care meets advanced analytics</div>
</section>
<section class="about" id="about">
    <div><h2>About Us</h2></div>
    <div class="sentence">MedDetect helps you explore potential health concerns through symptom analysis...</div>
</section>
<div class="container1">
    <section class="services">
        <div class="service-text">
            <h1>Our Services</h1>
            <p>Connect with AI Diagnosis for accurate disease predictions based on symptoms.</p>
            <h2>Disease Prediction</h2>
        </div>
    </section>    
</div>
"""

st.markdown(html_content, unsafe_allow_html=True)

# Symptom Input and Prediction
text_content = st.text_input("Enter your Symptoms (comma-separated):")
class_btn = st.button("Detect")

if class_btn:
    if not text_content:
        st.write("Invalid command, please enter symptoms.")
    else:
        with st.spinner('Model working....'):
            try:
                predictions = predictDisease(text_content)
                st.success(f"Prediction: {predictions}")
            except Exception as e:
                st.error(f"An error occurred: {e}")

html_content1 = """
<section id="contact">
    <div class="container-contact">
        <div class="contact-message">
            <h2>Contact Us</h2>
            <p>Feel free to contact us for any questions, feedback, or inquiries, and we'll respond soon.</p>
        </div>
        <div class="contact-form">
            <form action="#" method="POST">
                <div class="form-group">
                    <div class="half1">
                        <input type="text" name="first_name" placeholder="First Name" required>
                    </div>
                    <div class="half2">
                        <input type="text" name="last_name" placeholder="Last Name" required>
                    </div>
                </div>
                <div class="form-group">
                    <input type="email" name="email" placeholder="Your Email" required>
                </div>
                <div class="form-group">
                    <textarea name="message" placeholder="Your Message" rows="5" required></textarea>
                </div>
                <div class="form-group">
                    <button type="submit">Submit</button>
                </div>
            </form>
        </div>
    </div>
</section>
<footer>
    <div class="footer">
        <div class="footer-text">
            <p>Copyright &copy; 2023 MedDetect | All Rights Reserved.</p>
        </div>
    </div>
</footer>
"""

st.markdown(html_content1, unsafe_allow_html=True)
