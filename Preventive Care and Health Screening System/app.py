import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import random
import os

# Load the dataset
try:
    df = pd.read_csv("chronic_disease_management.csv")
except Exception as e:
    st.error(f"Error loading dataset: {str(e)}")

# Data Preprocessing
def preprocess_data(df):
    try:
        df['diagnosis_date'] = pd.to_datetime(df.get('diagnosis_date', None), errors='coerce')
        df['last_checkup_date'] = pd.to_datetime(df.get('last_checkup_date', None), errors='coerce')
        df['follow_up_schedule'] = pd.to_datetime(df.get('follow_up_schedule', None), errors='coerce')
        df['risk_score'] = df.get('quality_of_life_score', 0) / 10  # Example risk computation
        df['is_overdue'] = df['follow_up_schedule'] < pd.Timestamp.today()
        return df
    except Exception as e:
        st.error(f"Error in preprocessing data: {str(e)}")
        return df

df = preprocess_data(df)

# Function to load the BioBERT model safely
def load_biobert_model():
    model_name = "dmis-lab/biobert-base-cased-v1.1"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)  # Binary classification
        return tokenizer, model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

tokenizer, model = load_biobert_model()

# Function: Predict Risk
def predict_risk(patient_row):
    if not tokenizer or not model:
        return "Model not loaded"
    input_text = f"Patient data: {patient_row['chronic_disease']}, {patient_row['current_medications']}, {patient_row['lifestyle_interventions']}."
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=1)
    risk_class = torch.argmax(probs, dim=1).item()
    return "High Risk" if risk_class == 1 else "Low Risk"

# Function: Send Automated Emails
def send_automated_email(patient_id, is_overdue, risk_status, patient_name, patient_email):
    try:
        # SMTP Setup (using a free Gmail account here)
        smtp_server = 'smtp.gmail.com'
        smtp_port = 587
        sender_email = "your_email@gmail.com"  # Replace with your email
        sender_password = "your_email_password"  # Replace with your email password

        # Message Content
        message = MIMEMultipart()
        message['From'] = sender_email
        message['To'] = patient_email
        message['Subject'] = "Reminder: Preventive Care and Screening"

        body = f"Dear {patient_name},\n\n"
        if is_overdue:
            body += f"Your follow-up screening is overdue. Please schedule it as soon as possible.\n"
        body += f"Risk Status: {risk_status}\n\nPlease take the necessary actions to stay healthy."

        # Attach the message to the email
        message.attach(MIMEText(body, 'plain'))

        # Send the email using SMTP
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()  # Secure the connection
            server.login(sender_email, sender_password)
            text = message.as_string()
            server.sendmail(sender_email, patient_email, text)
        return "Email sent successfully!"
    except Exception as e:
        return f"Failed to send email: {str(e)}"

# Function: AI-Powered Symptom Checker
def symptom_checker(symptoms):
    try:
        # Simple logic to simulate symptom checking and recommendation
        known_symptoms = {
            "fever": "You may have a fever. Stay hydrated and monitor your temperature.",
            "headache": "You may have a headache. Consider taking over-the-counter pain relievers.",
            "cough": "You might have a cough. If persistent, consider seeing a doctor.",
            "fatigue": "Fatigue could be a sign of stress or illness. Ensure proper rest and hydration.",
            "shortness of breath": "Shortness of breath can be serious. Please consult a healthcare provider immediately.",
        }

        # Lowercasing symptoms for comparison
        symptoms = symptoms.lower().split(",")
        recommendations = []

        for symptom in symptoms:
            symptom = symptom.strip()
            if symptom in known_symptoms:
                recommendations.append(known_symptoms[symptom])
            else:
                recommendations.append(f"Symptom '{symptom}' is not recognized. Please consult a doctor for advice.")
        
        # Return a random recommendation if no exact match is found
        if not recommendations:
            recommendations = [random.choice(list(known_symptoms.values()))]

        return " ".join(recommendations)
    except Exception as e:
        st.error(f"Error in symptom checker: {str(e)}")
        return "Error checking symptoms."

# Streamlit Interface with Tabs
st.title("Preventive Care and Health Screening System")
tabs = st.tabs(["Dashboard", "Patient Search", "Data Insights", "Symptom Checker", "Emergency Alerts", "Segmentation", "Export Data"])

# Tab: Dashboard (Enhanced with Row-wise Display and Visualizations)
with tabs[0]:
    try:
        st.subheader("Overview of Patient Data and Visual Insights")

        # Filters
        age_range = st.slider("Filter by Age Range", int(df['age'].min()), int(df['age'].max()), (20, 60))
        selected_gender = st.selectbox("Filter by Gender", options=["All", "Male", "Female"])
        filtered_data = df[
            (df['age'].between(age_range[0], age_range[1])) & 
            ((df['gender'] == selected_gender) if selected_gender != "All" else True)
        ]
        
        # Display filtered data in a table
        st.write("### Filtered Patient Data")
        st.dataframe(filtered_data[['patient_id', 'patient_name', 'age', 'gender', 'risk_score', 'chronic_disease', 'current_medications']])
        
       
    except Exception as e:
        st.error(f"Error in dashboard section: {str(e)}")

# Tab: Patient Search
with tabs[1]:
    try:
        st.subheader("Search Patient by ID")
        patient_id_input = st.text_input("Enter Patient ID (partial or full):")
        if patient_id_input:
            matched_patients = df[df['patient_id'].astype(str).str.contains(str(patient_id_input))]
            if not matched_patients.empty:
                for _, patient_row in matched_patients.iterrows():
                    st.write(f"### Patient Details (ID: {patient_row['patient_id']})")
                    st.write(f"**Name:** {patient_row['patient_name']}")
                    st.write(f"**Age:** {patient_row['age']}")
                    st.write(f"**Gender:** {patient_row['gender']}")
                    st.write(f"**Chronic Disease:** {patient_row['chronic_disease']}")
                    st.write(f"**Current Medications:** {patient_row['current_medications']}")
                    st.write(f"**Risk Score:** {patient_row['risk_score']}")
                    st.write(f"**Follow-up Schedule:** {patient_row['follow_up_schedule']}")
                    st.write(f"**Overdue:** {patient_row['is_overdue']}")
                    st.write(f"**Risk Status:** {predict_risk(patient_row)}")
                    if patient_row['is_overdue']:
                        st.warning("This patient is overdue for a follow-up!")
                        if st.button(f"Send Reminder to Patient {patient_row['patient_id']}", key=f"send_reminder_{patient_row['patient_id']}"):
                            message_status = send_automated_email(patient_row['patient_id'], patient_row['is_overdue'], predict_risk(patient_row), patient_row['patient_name'], patient_row['email'])
                            st.success(message_status)
            else:
                st.error("No matching patients found!")
    except Exception as e:
        st.error(f"Error in patient search: {str(e)}")
        
 #Tab:  Data Insights (Charts and Visualizations)

        st.write("### Data Insights")
        fig = plt.figure(figsize=(10, 6))
        sns.histplot(df['risk_score'], kde=True)
        st.pyplot(fig)
        
        st.write("### Risk Score Distribution")
        fig2 = plt.figure(figsize=(10, 6))
        sns.boxplot(x='gender', y='risk_score', data=df)
        st.pyplot(fig2)

# Tab: Symptom Checker
with tabs[3]:
    try:
        st.subheader("Symptom Checker")
        symptoms_input = st.text_input("Enter Symptoms (comma separated):")
        if symptoms_input:
            recommendations = symptom_checker(symptoms_input)
            st.write("### Recommendations:")
            st.write(recommendations)
    except Exception as e:
        st.error(f"Error in symptom checker: {str(e)}")

# Tab: Emergency Alerts
with tabs[4]:
    try:
        st.subheader("Send Emergency Alerts")
        alert_patient_id = st.selectbox("Select Patient ID", df['patient_id'].values)
        alert_patient = df[df['patient_id'] == alert_patient_id].iloc[0]
        if alert_patient:
            st.write(f"### Patient: {alert_patient['patient_name']}")
            alert_status = st.radio("Send Alert for Emergency", ["No", "Yes"])
            if alert_status == "Yes":
                emergency_message = f"Emergency alert for patient {alert_patient['patient_name']}: Risk status is critical."
                st.write(emergency_message)
                # Send Email to patient or caretaker
                message_status = send_automated_email(alert_patient['patient_id'], alert_patient['is_overdue'], "Critical", alert_patient['patient_name'], alert_patient['email'])
                st.success(message_status)
    except Exception as e:
        st.error(f"Error in emergency alerts section: {str(e)}")

# Tab: Segmentation
with tabs[5]:
    try:
        st.subheader("Patient Segmentation")
        st.write("Segmentation by Risk Score")
        risk_labels = df['risk_score'].apply(lambda x: 'High' if x >= 0.7 else 'Low')
        df['Risk Label'] = risk_labels
        st.write(df[['patient_id', 'patient_name', 'Risk Label']].groupby('Risk Label').count())
    except Exception as e:
        st.error(f"Error in segmentation: {str(e)}")

# Tab: Export Data
with tabs[6]:
    st.write("Export Data as CSV")
    st.download_button("Download CSV", df.to_csv(index=False), "patient_data.csv", "text/csv")
