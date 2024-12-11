# Preventive Care and Health Screening System

This project is a **Preventive Care and Health Screening System** developed using Python. It combines Natural Language Processing (NLP) and Machine Learning techniques to assist healthcare providers in managing patient screenings, tracking risk factors, and sending notifications or reminders. The system leverages **Streamlit** for a user-friendly interface and **BioBERT** for AI-powered risk analysis.

## Features

1. **Data Dashboard**:
   - Visualize and filter patient data by age range and gender.
   - View risk scores and chronic disease data.

2. **Patient Search**:
   - Search for patients by ID and view their details.
   - Get AI-predicted risk status and overdue follow-up alerts.

3. **Symptom Checker**:
   - Enter symptoms to get AI-powered recommendations for care.

4. **Automated Notifications**:
   - Send email reminders to patients about overdue screenings.
   - Generate emergency alerts for critical risk cases.

5. **Data Insights**:
   - Generate risk score histograms and gender-based risk score boxplots.

6. **Patient Segmentation**:
   - Categorize patients into high-risk and low-risk groups.

7. **AI-Powered Risk Prediction**:
   - Utilize BioBERT for binary risk classification.

## Tech Stack

- **Python**: Core programming language.
- **Streamlit**: Interactive web application framework.
- **Pandas**: Data manipulation and preprocessing.
- **Seaborn & Matplotlib**: Data visualization.
- **Transformers**: NLP model loading (BioBERT).
- **Torch**: Backend for BioBERT inference.
- **SMTP (smtplib)**: Email automation.

## Setup and Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/preventive-care-system.git
   cd preventive-care-system
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download and prepare the dataset:
   - Ensure the dataset file `chronic_disease_management.csv` is in the project directory.

4. Configure email settings:
   - Replace `your_email@gmail.com` and `your_email_password` in the script with valid credentials.

5. Run the application:
   ```bash
   streamlit run app.py
   ```

## Usage

- Open the Streamlit app in your browser.
- Navigate through tabs to explore patient data, check symptoms, or send notifications.
- Visualize data insights and segment patients based on risk scores.

## Key Functions

### `preprocess_data(df)`
Processes and prepares the dataset, handling date conversions and computing risk scores.

### `load_biobert_model()`
Loads the BioBERT model and tokenizer for risk classification.

### `predict_risk(patient_row)`
Predicts the risk level for a patient based on their data using the BioBERT model.

### `send_automated_email(patient_id, is_overdue, risk_status, patient_name, patient_email)`
Sends personalized email notifications to patients regarding their health status.

### `symptom_checker(symptoms)`
Provides recommendations based on input symptoms.

## Example Visualizations

1. **Risk Score Distribution**:
   - Histogram and boxplot to visualize the spread of risk scores.
2. **Patient Segmentation**:
   - Tabular display of patient counts in high-risk and low-risk categories.

## Future Enhancements

- Add support for additional languages in the symptom checker.
- Integrate patient feedback on email notifications.
- Expand AI-powered analysis to include multi-class classification.
