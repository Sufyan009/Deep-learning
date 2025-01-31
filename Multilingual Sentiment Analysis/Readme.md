# Multilingual Sentiment Analysis

## Overview
The **Multilingual Sentiment Analysis** project leverages natural language processing (NLP) techniques to classify multilingual product reviews into sentiment categories: **Positive**, **Neutral**, and **Negative**. The project includes a Python-based pipeline that uses Hugging Face's transformers, a pre-trained multilingual model (`xlm-roberta-base`), and a Streamlit app for real-time sentiment predictions.

This system supports multiple languages and provides sentiment analysis on user-submitted reviews, allowing businesses and customers to understand product feedback in various languages.

## Features
- **Multilingual Support:** The system supports multiple languages, including English, Spanish, German, French, Japanese, and Chinese.
- **Sentiment Classification:** Classifies product reviews into three sentiment categories: Positive, Neutral, and Negative.
- **Streamlit Interface:** Provides an interactive interface to input text and receive real-time sentiment predictions.
- **Automatic Language Detection:** Detects the language of the review before making predictions.
- **Tokenization and Model Deployment:** Utilizes Hugging Face's transformer models for tokenization and sentiment prediction.

## Dataset
The model is trained using the **Multilingual Amazon Reviews Corpus**. Reviews are classified based on their star ratings:
- **Positive:** Star ratings 4 and 5
- **Neutral:** Star rating 3
- **Negative:** Star ratings 1 and 2

## Project Structure
The project consists of the following main components:
1. **Streamlit Application (`app.py`)**: A web interface for users to input product reviews and view the sentiment prediction.
2. **Model Training Notebook (`model_training.ipynb`)**: A Jupyter notebook where the sentiment analysis model is trained on the dataset.
3. **Sentiment Model (`sentiment_model`)**: The saved model and tokenizer after training.

## Installation & Setup

### Requirements
Make sure you have the following libraries installed:
- `streamlit`
- `transformers`
- `torch`
- `langdetect`
- `datasets`
- `sklearn`
- `regex`
- `pandas`

You can install the dependencies by running the following command:
```bash
pip install -r requirements.txt
```

### Model Training
To train the model:
1. Download the **Multilingual Amazon Reviews Corpus** and preprocess it.
2. Run the `model_training.ipynb` notebook to preprocess the dataset, train the model, and save the model to disk.
3. The trained model will be saved in the `sentiment_model` directory.

### Running the Streamlit App
Once the model is trained, you can run the Streamlit app to interact with the model:
```bash
streamlit run app.py
```
This will launch a web app where you can input a product review and receive a sentiment prediction.

## How to Use
1. **Input Review**: Enter a product review in any language into the provided text area.
2. **Prediction**: Click the "Predict Sentiment" button to see the sentiment prediction: Positive, Neutral, or Negative.
3. **Example Reviews**: You can also explore example reviews listed in the sidebar.

## Performance Evaluation
After training the model, you can evaluate its performance on the test set using the following metrics:
- **Accuracy**
- **Macro F1-Score**

Results for the final evaluation will be displayed after running the training script.

## Model Results

The model achieves the following results on the test set:

- **Accuracy**: 76.2%
- **Macro F1-Score**: 0.6986

These results demonstrate the model's ability to classify product reviews into sentiment categories with a good balance between precision and recall.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements
- **Hugging Face Transformers**: For providing pre-trained models and tokenizers.
- **Streamlit**: For the easy-to-use web application framework.
- **Multilingual Amazon Reviews Corpus**: The dataset used for training the sentiment analysis model.
