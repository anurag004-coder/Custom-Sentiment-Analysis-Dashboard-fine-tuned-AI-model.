Project Overview
This project is an interactive, web-based Natural Language Processing (NLP) dashboard designed to perform real-time sentiment classification. It leverages a custom fine-tuned Transformer model hosted on the Hugging Face Hub, allowing users to evaluate the emotional tone of text seamlessly through a Streamlit interface.
Live Demo
🚀 https://custom-sentiment-analysis-dashboard-fine-tuned-ai-model-pkgxql.streamlit.app/

Features
Real-Time Inference: Instantly analyze single sentences and paragraphs for positive or negative sentiment.
Confidence Scoring: Provides percentage-based confidence metrics for every prediction to gauge model certainty.
Batch Processing: Supports uploading CSV/Excel datasets for bulk sentiment analysis (if enabled in the sidebar).
Cloud Deployment: Fully deployed via Streamlit Community Cloud with seamless integration to the Hugging Face Model Hub.

Tech Stack
Language: Python
Frontend/Deployment: Streamlit, Streamlit Community Cloud
Machine Learning: Hugging Face transformers, PyTorch/TensorFlow
Version Control: Git, GitHub

Local Installation
To run this project locally on your machine:

Clone the repository:
  git clone https://github.com/your-username/Custom-Sentiment-Analysis-Dashboard-fine-tuned-AI-model.git
cd Custom-Sentiment-Analysis-Dashboard-fine-tuned-AI-model

Install dependencies:
  pip install -r requirements.txt

Run the application:
  streamlit run app.py

Model Architecture
The underlying model is a fine-tuned Transformer architecture optimized for sequence classification. By migrating the model weights to the Hugging Face Hub, this application maintains a lightweight local footprint while delivering high-accuracy NLP predictions.
