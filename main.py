import streamlit as st
import pickle
import numpy as np
import time
import warnings

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

symptom_index = {}
for index, value in enumerate(symptoms):
	symptom = " ".join([i.capitalize() for i in value.split("_")])
	symptom_index[symptom] = index

data_dict = {
	"symptom_index":symptom_index,
	"predictions_classes":diseases
}

def predictDisease(symptoms):
    with open('final_rf_model.pkl', 'rb') as f:
        model = pickle.load(f)
    symptoms = symptoms.split(",")

    input_data = [0] * len(data_dict["symptom_index"])
    for symptom in symptoms:
        index = data_dict["symptom_index"][symptom.strip()]
        input_data[index] = 1

    input_data = np.array(input_data).reshape(1, -1)

    rf_prediction = data_dict["predictions_classes"][model.predict(input_data)[0]]
  
    return rf_prediction



st.set_page_config(layout="wide")

# Header
st.markdown('<header>'
                '<div class="container">'
                '<div class="logo">'
                '<h2>MedDetect</h2>'
                '</div>'
                '<nav class="nav-a">'
                '<ul>'
                '<li><a href="home.html">Home</a></li>'
                '<li><a href="#about">About</a></li>'
                '<li><a href="#contact">Contact</a></li>'
                '</ul>'
                '</nav>'
                '</div>'
                '</header>', unsafe_allow_html=True)

#Header Css
st.markdown("""

        <style>
            /* styles.css */

/* Reset default browser styles */
body, h1, h2, h3, p, ul, li {
    margin: 0;
    padding: 0;
}

/* Header styles */
header {
    background-color: #D2E3C8;
    padding: 2rem 0;
}

.container {
    width: 80%;
    margin: 0 auto;
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.logo {
    display: flex;
    align-items: center;
}

.logo h2 {
    font-size: 24px;
    font-weight: bold;
    color: #1b4e38;
}

.nav-a ul {
    list-style: none;
    display: flex;
}

.nav-a li {
    margin-left: 20px;
}

.nav-a li a {
    text-decoration: none;
    color: #1b4e38;
    font-size: 18px;
    font-weight: bold;
}

.nav-a li a:hover {
    color: #4a8d70;
}

</style>
""", unsafe_allow_html=True
    )

# About Us Section
st.markdown('<section class="about" id="about">'
                '<div class="sentence">MedDetect helps you explore potential health concerns through symptom'
                ' analysis. Get insightful suggestions and basic information, but remember, professional diagnosis and'
                ' treatment are crucial. Seek expert advice for optimal health. Browse our website to learn more about us.'
                '</div>'
                '</section>', unsafe_allow_html=True)

# Diagnosis Section
st.markdown('<section class="services">'
                '<div class="service-text">'
                '<p>Connect with AI Diagnosis for accurate disease predictions based on symptoms.</p>'
                '</div>'
                '</section>', unsafe_allow_html=True)



st.title('Disease Prediction')
st.markdown("Prediction From user Symptoms")
text_content=st.text_input("Enter your Symptoms")
st.code(text_content)

# file_uploaded = st.file_uploader("Choose File", type=["txt", "exe"])
  # Detection Button
class_btn = st.button("Detect")

if class_btn:
    if text_content is None:
        st.write("Invalid command, please upload a valid Text or Executable file")
    else:
        with st.spinner('Model working....'):
            predictions = predictDisease(text_content)
            time.sleep(1)
            st.success(predictions)

   # Contact Us Section
st.markdown('<section id="contact">'
                '<div class="container-contact">'
                '<div class="contact-message">'
                '<h2>Contact Us</h2>'
                '<p>Feel free to contact us for any questions, feedback, or inquiries, and we\'ll respond soon.</p>'
                '</div>'
                '<div class="contact-form">'
                '<form action="#" method="POST">'
                '<div class="form-group">'
                '<div class="half1">'
                '<input type="text" name="first_name" placeholder="First Name" required>'
                '</div>'
                '<div class="half2">'
                '<input type="text" name="last_name" placeholder="Last Name" required>'
                '</div>'
                '</div>'
                '<div class="form-group">'
                '<input type="email" name="email" placeholder="Your Email" required>'
                '</div>'
                '<div class="form-group">'
                '<textarea name="message" placeholder="Your Message" rows="5" required></textarea>'
                '</div>'
                '<div class="form-group">'
                '<button type="submit">Submit</button>'
                '</div>'
                '</form>'
                '</div>'
                '</div>'
                '</section>', unsafe_allow_html=True)


# Footer
st.markdown("---")
st.markdown('<footer>'
                '<div class="footer">'
                '<div class="footer-text">'
                '<p>Copyright &copy; 2023 MedDetect | All Rights Reserved.</p>'
                '</div>'
                '</div>'
                '</footer>', unsafe_allow_html=True)


st.markdown("""
        <style>
            #contact {
                background-color: #DDDDDD;
                padding: 50px 0;
            }

            .container-contact {
                max-width: 800px;
                margin: 0 auto;
                padding: 20px;
                display: flex;
            }

            .contact-message,
            .contact-form {
                flex: 1;
            }

            .contact-message h2 {
                color: #333;
            }

            .contact-message p {
                color: #666;
            }

            .contact-form form {
                display: flex;
                flex-direction: column;
            }

            .form-group {
                margin-bottom: 20px;
                display: flex;
            }

            .half1,
            .half2 {
                flex: 1;
            }

            .half2 {
                margin-left: 10px;
            }

            input[type="text"],
            input[type="email"],
            textarea {
                width: 100%;
                padding: 10px;
                border: 1px solid #ccc;
                border-radius: 5px;
            }

            textarea {
                resize: vertical;
            }

            button {
                background-color: #4CAF50;
                color: white;
                padding: 10px 20px;
                border: none;
                border-radius: 5px;
                cursor: pointer;
            }

            button:hover {
                background-color: #45a049;
            }
        </style>
    """, unsafe_allow_html=True)
