import streamlit as st
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from fpdf import FPDF
import matplotlib.pyplot as plt
import io
#import base64
import tempfile

# Load the trained model and scaler
with open('D:/Guvi/Own_Projects/Diabetes_Prediction/Diabetes_Prediction_Model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Load the preprocessed dataset to fit the scaler
data = pd.read_csv('D:/Guvi/Own_Projects/Diabetes_Prediction/preprocessed_diabetes_data.csv')

# Initialize the scaler
scaler = StandardScaler()
# Fit the scaler on numerical features
numerical_features = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']
scaler.fit(data[numerical_features])

# Streamlit application for Diabetes Prediction
def main():
    st.title("Diabetes Prediction App")

    # User inputs
    name = st.text_input("Enter Name")
    gender = st.selectbox("Gender", ("Male", "Female"))
    age = st.number_input("Age", min_value=0.0, max_value=120.0, value=30.0, step=1.0)
    hypertension = st.selectbox("Hypertension", ("No", "Yes"))
    heart_disease = st.selectbox("Heart Disease", ("No", "Yes"))
    smoking_history = st.selectbox("Smoking History", ("never", "current", "formerly", "No Info", "ever", "not current"))
    bmi = st.number_input("BMI", min_value=10.0, max_value=100.0, value=25.0, step=0.1)
    HbA1c_level = st.number_input("HbA1c Level", min_value=3.5, max_value=9.0, value=5.5, step=0.1)
    blood_glucose_level = st.number_input("Blood Glucose Level", min_value=80, max_value=300, value=140, step=1)

    # Convert categorical inputs to numeric
    gender_numeric = 1 if gender == "Male" else 0
    hypertension_numeric = 1 if hypertension == "Yes" else 0
    heart_disease_numeric = 1 if heart_disease == "Yes" else 0
    smoking_history_numeric = {
        "never": 0,
        "current": 1,
        "formerly": 2,
        "No Info": 3,
        "ever": 4,
        "not current": 5
    }[smoking_history]

    # Create feature vector
    inputs = np.array([[age, bmi, HbA1c_level, blood_glucose_level]])
    scaled_inputs = scaler.transform(inputs)
    feature_vector = np.concatenate(([gender_numeric, hypertension_numeric, heart_disease_numeric, smoking_history_numeric], scaled_inputs.flatten())).reshape(1, -1)

    # Prediction
    if st.button("Predict"):
        prediction = model.predict(feature_vector)
        result = "Diabetic" if prediction[0] == 1 else "Non-Diabetic"
        name_prefix = "Mr." if gender == "Male" else "Ms." if gender == "Female" else ""

        # Display the medical report
        st.markdown(f"### Medical Report for {name_prefix} {name}")
        st.markdown(
            f"""
            **Patient Name:** {name_prefix} {name}  
            **Gender:** {gender}  
            **Age:** {age}  
            **Hypertension:** {hypertension}  
            **Heart Disease:** {heart_disease}  
            **Smoking History:** {smoking_history}  
            **BMI:** {bmi}  
            **HbA1c Level:** {HbA1c_level}  
            **Blood Glucose Level:** {blood_glucose_level}  
            """, unsafe_allow_html=True
        )

        # Highlight the prediction result
        st.markdown(f"### <span style='color:maroon;'>Prediction: {result}</span>", unsafe_allow_html=True)
        
        # Print the personalized message
        if result == "Diabetic":
            message = "Take Care of your Health, Have a NICE Day"
        else:
            message = "Congrats! You seem to be Healthy, Have a NICE Day"

        st.markdown(f"#### {message}", unsafe_allow_html=True)

        # Generate and display PDF
        pdf_buffer = generate_pdf(name, gender, age, hypertension, heart_disease, smoking_history, bmi, HbA1c_level, blood_glucose_level, result)
        st.download_button(label="Download PDF", data=pdf_buffer, file_name='Medical_Report.pdf', mime='application/pdf')

        # Generate and display Image
        img_buffer = generate_image(name, gender, age, hypertension, heart_disease, smoking_history, bmi, HbA1c_level, blood_glucose_level, result)
        st.download_button(label="Download Image", data=img_buffer, file_name='Medical_Report.png', mime='image/png')

def generate_pdf(name, gender, age, hypertension, heart_disease, smoking_history, bmi, HbA1c_level, blood_glucose_level, result):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    name_prefix = "Mr." if gender == "Male" else "Ms." if gender == "Female" else ""
    pdf.set_font("Arial", size=16)
    pdf.cell(200, 10, txt=f"Medical Report for {name_prefix} {name}", ln=True, align='C')
    pdf.ln(10)
    
    report_data = [
        ("Patient Name", f"{name_prefix} {name}"),
        ("Gender", gender),
        ("Age", age),
        ("Hypertension", hypertension),
        ("Heart Disease", heart_disease),
        ("Smoking History", smoking_history),
        ("BMI", bmi),
        ("HbA1c Level", HbA1c_level),
        ("Blood Glucose Level", blood_glucose_level),
        ("Prediction", result)
    ]
    
    pdf.set_font("Arial", size=12)
    for key, value in report_data:
        pdf.cell(200, 10, txt=f"{key}: {value}", ln=True, align='L')
    
    # Save PDF to a temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
    pdf.output(temp_file.name)
    
    # Read the file into a BytesIO object
    pdf_buffer = io.BytesIO()
    with open(temp_file.name, 'rb') as f:
        pdf_buffer.write(f.read())
    pdf_buffer.seek(0)
    
    # Clean up the temporary file
    temp_file.close()
    
    return pdf_buffer.getvalue()

def generate_image(name, gender, age, hypertension, heart_disease, smoking_history, bmi, HbA1c_level, blood_glucose_level, result):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.axis('off')
    
    name_prefix = "Mr." if gender == "Male" else "Ms." if gender == "Female" else ""
    report_text = (
        f"Medical Report for {name_prefix} {name}\n\n"
        f"Patient Name: {name_prefix} {name}\n"
        f"Gender: {gender}\n"
        f"Age: {age}\n"
        f"Hypertension: {hypertension}\n"
        f"Heart Disease: {heart_disease}\n"
        f"Smoking History: {smoking_history}\n"
        f"BMI: {bmi}\n"
        f"HbA1c Level: {HbA1c_level}\n"
        f"Blood Glucose Level: {blood_glucose_level}\n\n"
        f"Prediction: {result}"
    )
    
    plt.text(0.5, 0.95, report_text, fontsize=12, ha='center', va='top', color='black', wrap=True)
    plt.text(0.5, 0.1, f"{result} - {'Take Care of your Health, Have a NICE Day' if result == 'Diabetic' else 'Congrats! You seem to be Healthy, Have a NICE Day'}", 
             fontsize=14, ha='center', va='top', color='maroon', wrap=True)
    plt.tight_layout()
    
    # Save the image to a BytesIO object
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png')
    img_buffer.seek(0)
    
    return img_buffer.getvalue()

if __name__ == "__main__":
    main()
