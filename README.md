**Disease Prediction using Machine Learning**

This project integrates predictive algorithms for health prognosis, enabling accurate disease prediction and enhancing proactive healthcare management. It also provides a LangChain-powered API in Google Colab that converts free-text problem descriptions into structured symptoms for automated disease prediction.

**🧠 Table of Contents**

-> Features

-> Project Structure

-> Tech Stack

-> Setup & Installation

-> Usage

-> Model Training & Deployment

-> LangChain API for Symptom Extraction


**Features**

-> Predictive algorithms: Builds machine learning models to predict potential diseases based on user data.

-> Web Interface: UI built with HTML, CSS, JavaScript for user-friendly interaction.

-> Model Serialization: Uses Pickle to save & load trained models.

-> Interactive UI: Optionally using Streamlit for real-time interaction and demo.

-> Testing & CSV Inputs: Supports test cases via CSV files; has scripts for prediction.

-> LangChain Integration: Converts natural problem descriptions into structured symptom lists using Google Gemini API.

**Project Structure**

Disease-Prediction-using-Machine-Learning/

├── .devcontainer/ → Dev environment config files

├── Training2.csv → Training dataset

├── Testing3.csv → Testing / validation dataset

├── final_rf_model.pkl → Trained Random Forest model

├── Project1.ipynb → Notebook for experimentation / model building

├── test.py / testt2.py → Scripts for running predictions

├── home.js / home.css / header.css → Front-end JS & CSS resources

├── first.jpeg / second.webp → Images / assets

├── requirements.txt → Python dependencies

└── README.md → Project documentation

**Tech Stack**

-> Languages: Python, JavaScript, HTML, CSS

-> ML Libraries: scikit-learn, pandas, numpy

-> LangChain: For LLM-powered symptom extraction

-> LLM: Google Gemini Pro API

-> Frameworks: Streamlit, Flask, Flask-Ngrok

-> Data Format: CSV

-> Tools: Pickle, Jupyter Notebook

**Setup & Installation**

Clone the repository:
git clone https://github.com/adibirje14/Disease-Prediction-using-Machine-Learning.git

cd Disease-Prediction-using-Machine-Learning

(Optional) Create & activate a virtual environment:
python3 -m venv venv
source venv/bin/activate (On Windows: venv\Scripts\activate)

Install dependencies:
pip install -r requirements.txt

For Google Colab setup, install extra packages:
!pip install -q --upgrade google-generativeai langchain-google-genai chromadb pypdf
!pip install langchain
!pip install flask-ngrok

**Usage**

-> Jupyter Notebook: Open Project1.ipynb to explore data, training, and evaluation.

-> Prediction scripts: Run test.py or testt2.py to make predictions on given inputs or CSV data.

-> Front-end / UI: Use the HTML/JS/CSS files to build a simple interface for inputs and predictions.

-> Streamlit (if used): streamlit run test.py

**Model Training & Deployment**

-> The ML model (Random Forest) is trained using the training dataset.

-> Once satisfied with performance, the model is saved as final_rf_model.pkl.

-> The prediction scripts or UI load this model to make real-time predictions without retraining.

-> LangChain API for Symptom Extraction: We integrated LangChain with Google Gemini Pro API to process natural language health queries and automatically extract symptoms from a predefined list.

**How It Works**

- User enters a free-text health problem description (e.g., “My nose has been itchy for two days, I have a sore throat, and I started getting a headache this morning”).
  
- The system uses LangChain PromptTemplate and Gemini Pro LLM to match symptoms against a predefined medical symptom list.
  
- The API returns a JSON list of identified symptoms, which is then passed to the disease prediction model.

**Example Output**

Input Query:
"My nose has been itchy for the past two days and I have a sore throat too and I also started getting a headache this morning"

Extracted Symptoms (JSON):

{
  "symptoms": ["itching", "patches_in_throat", "headache"]
}

**Try It Yourself**

You can run and test the LangChain symptom extraction API directly on Google Colab using the link below:

https://colab.research.google.com/drive/1nWlKeX0_Zy-fsf9IU_Js1a_j1OOEtjbJ#scrollTo=Y0y93NBL6FNa

-> Open LangChain Symptom Extraction in Google Colab: This allows anyone to experiment with the natural language to structured symptom extraction workflow without setting up anything locally.

**Implemented by Aditya Birje**
