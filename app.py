import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# --- 1. Page Configuration ---
st.set_page_config(page_title="Sentiment Analysis Dashboard",
                   page_icon="📊", layout="wide")
st.title("📊 Custom Sentiment Analysis Dashboard")
st.markdown(
    "Analyze your original text data using your locally hosted fine-tuned AI model.")

# --- 2. Load Your Custom NLP Model (Custom PyTorch Implementation) ---


@st.cache_resource
def load_model():
    # Load the tokenizer and model directly instead of using the generic pipeline
    tokenizer = AutoTokenizer.from_pretrained("./custom_model")
    model = AutoModelForSequenceClassification.from_pretrained(
        "./custom_model")
    return tokenizer, model


try:
    tokenizer, model = load_model()
    st.sidebar.success("✅ Custom Model Loaded Successfully!")
except Exception as e:
    st.sidebar.error(f"❌ Error loading model: {e}")
    st.markdown("### ⚠️ Setup Required")
    st.info("Please make sure your fine-tuned model files are located inside the `custom_model` folder in your project directory.")
    st.stop()

# --- 3. Custom Inference Engine (The Bug Fix) ---


def predict_sentiment(text):
    # 1. Tokenize the text
    inputs = tokenizer(text, return_tensors="pt",
                       truncation=True, max_length=512)

    # 🛑 THE FIX: Intercept and delete token_type_ids before they crash the model
    if "token_type_ids" in inputs:
        del inputs["token_type_ids"]

    # 2. Pass to the model safely
    with torch.no_grad():
        outputs = model(**inputs)

    # 3. Calculate probabilities and confidence scores
    logits = outputs.logits
    predicted_class_id = logits.argmax().item()
    confidence = torch.nn.functional.softmax(
        logits, dim=-1)[0][predicted_class_id].item()

    raw_label = model.config.id2label[predicted_class_id]

    if raw_label in ["POSITIVE", "LABEL_1"]:
        return "Positive", confidence
    else:
        return "Negative", confidence


# --- 4. Sidebar Options ---
st.sidebar.markdown("### ⚙️ Analysis Settings")
analysis_mode = st.sidebar.radio(
    "Select Input Type:", ["Upload Dataset (CSV/Excel)", "Analyze Single Text"])

# --- 5. Mode A: Upload Dataset ---
if analysis_mode == "Upload Dataset (CSV/Excel)":
    st.markdown("### 📁 Batch Dataset Processing")
    uploaded_file = st.file_uploader(
        "Choose a CSV or Excel file containing text data", type=["csv", "xlsx"])

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
        except Exception as e:
            st.error(f"Error reading file: {e}")
            st.stop()

        st.write("📋 **Preview of Uploaded Data:**", df.head(5))
        text_column = st.selectbox(
            "Select the column containing the text strings:", df.columns)

        if st.button("🚀 Process Entire Dataset", type="primary"):
            results_records = []
            pos_count = 0
            neg_count = 0

            progress_bar = st.progress(0)
            status_text = st.empty()
            total_rows = len(df)

            for idx, row in df.iterrows():
                text_item = row[text_column]

                if pd.isna(text_item):
                    continue
                text_str = str(text_item).strip()
                if text_str == "":
                    continue

                try:
                    # Call our newly built custom PyTorch function!
                    label_display, score = predict_sentiment(text_str)

                    if label_display == "Positive":
                        pos_count += 1
                    else:
                        neg_count += 1

                    results_records.append({
                        "Text Data": text_str,
                        "Predicted Sentiment": label_display,
                        "Confidence": f"{score:.2%}"
                    })
                except Exception as inference_error:
                    st.error(
                        f"❌ Error parsing row {idx + 1}: {inference_error}")
                    continue

                progress_bar.progress((idx + 1) / total_rows)
                status_text.text(f"Analyzing row {idx + 1} of {total_rows}...")

            status_text.empty()
            progress_bar.empty()

            if results_records:
                st.success(
                    f"📊 Completed! Successfully analyzed {len(results_records)} rows.")

                metric_cols = st.columns(2)
                metric_cols[0].metric("Total Positive Items", pos_count)
                metric_cols[1].metric("Total Negative Items", neg_count)

                fig = go.Figure(data=[go.Bar(
                    x=['Positive', 'Negative'],
                    y=[pos_count, neg_count],
                    marker_color=['#2ecc71', '#e74c3c']
                )])
                fig.update_layout(title="Dataset Sentiment Breakdown",
                                  template="plotly_white", height=350)
                st.plotly_chart(fig, use_container_width=True)

                st.markdown("### 📋 Complete Analysis Breakdown")
                output_df = pd.DataFrame(results_records)
                st.dataframe(output_df, use_container_width=True,
                             hide_index=True)
            else:
                st.warning("No rows were successfully processed.")

# --- 6. Mode B: Analyze Single Text ---
else:
    st.markdown("### ✍️ Single Sentence Playground")
    user_input = st.text_area("Type or paste custom text snippet to evaluate below:",
                              placeholder="Type a sentence here...")

    if st.button("🔍 Run Evaluation", type="primary"):
        if user_input.strip() != "":
            with st.spinner("Running AI inference..."):
                try:
                    label_display, score = predict_sentiment(user_input)

                    if label_display == "Positive":
                        st.success(
                            f"**Result:** Positive Sentiment (Confidence: {score:.2%})")
                    else:
                        st.error(
                            f"**Result:** Negative Sentiment (Confidence: {score:.2%})")
                except Exception as e:
                    st.error(f"Inference error: {e}")
        else:
            st.warning("Please type some valid text characters first.")
