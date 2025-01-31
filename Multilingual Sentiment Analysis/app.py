import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from langdetect import detect, LangDetectException

# Define the model path
model_path = r"D:\Devsinc\Multilingual Sentiment Analysis\sentiment_models"

# Ensure model and tokenizer are loaded correctly
try:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    st.write("Tokenizer loaded successfully!")
except Exception as e:
    st.error(f"Error loading tokenizer: {e}")

try:
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    st.write("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")

# Define sentiment labels
id2label = {0: "Negative", 1: "Neutral", 2: "Positive"}
label2id = {"Negative": 0, "Neutral": 1, "Positive": 2}

# Ensure that the model is on the correct device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Tokenizer function to process the input
def tokenize_input(input_text):
    return tokenizer(
        input_text,
        padding='max_length',
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )

# Prediction function
def predict_sentiment(input_text):
    inputs = tokenize_input(input_text).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    prediction = torch.argmax(logits, dim=-1).item()
    sentiment = id2label[prediction]
    return sentiment

# Function to detect language
def detect_language(text):
    try:
        return detect(text)
    except LangDetectException:
        return None

# Streamlit app layout
st.title("Multilingual Sentiment Analysis")

st.write(
    "Enter a product review in any language, and the model will predict the sentiment (Positive, Neutral, or Negative)."
)

# Text input for review
review_text = st.text_area("Enter your review:", "Type here...")

# Button for prediction
if st.button("Predict Sentiment"):
    if review_text.strip():
        # Detect language of input text
        language = detect_language(review_text)
        if language is None:
            st.error("Could not detect the language of the input text.")
        else:
            with st.spinner("Predicting sentiment..."):
                sentiment = predict_sentiment(review_text)
                st.write(f"Sentiment: **{sentiment}** (Detected Language: {language.upper()})")
    else:
        st.warning("Please enter a review text.")

# Display example reviews for better understanding
st.sidebar.header("Example Reviews")
st.sidebar.write("Example 1: This product is amazing! 5 stars!")
st.sidebar.write("Example 2: The product is okay, but could be improved. 3 stars.")
st.sidebar.write("Example 3: I am very disappointed. It broke after a few uses. 1 star.")
st.sidebar.write("Example 4 (German): Dieses Produkt ist ausgezeichnet! 5 Sterne!")
st.sidebar.write("Example 5 (Italian): Il prodotto è ok, ma potrebbe essere migliorato. 3 stelle.")
st.sidebar.write("Example 6 (Portuguese): O produto é muito bom, mas precisa de melhorias. 4 estrelas.")
st.sidebar.write("Example 7 (Dutch): Ik ben erg teleurgesteld. Het brak na een paar keer gebruiken. 1 ster.")