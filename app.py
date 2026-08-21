import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch


st.set_page_config(
    page_title="Sentiment Analysis Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Custom Sentiment Analysis Dashboard")

st.markdown(
    """
    Analyze text using a custom fine-tuned Transformer sentiment
    classification model.
    """
)


MODEL_ID = "Anurag4848/sentiment-analysis-custom"


@st.cache_resource
def load_model():
    """
    Load the fine-tuned tokenizer and Transformer model
    from Hugging Face.
    """

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID
    )

    # Put model into evaluation mode
    model.eval()

    return tokenizer, model


try:
    tokenizer, model = load_model()

    st.sidebar.success(
        "✅ Fine-tuned model loaded successfully!"
    )

except Exception as e:
    st.sidebar.error(
        "❌ Unable to load the fine-tuned model."
    )

    st.markdown("### ⚠️ Model Setup Required")

    st.info(
        """
        Make sure your Hugging Face model repository exists and that
        MODEL_ID in app.py matches your repository.

        Example:

        username/sentiment-analysis-custom
        """
    )

    with st.expander("Show technical error"):
        st.code(str(e))

    st.stop()


# ============================================================
# 4. SENTIMENT PREDICTION FUNCTION
# ============================================================

def predict_sentiment(text):
    """
    Predict sentiment and return:

    - Sentiment label
    - Confidence score
    """

    # Tokenize input text
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    if "token_type_ids" in inputs:
        del inputs["token_type_ids"]

    # Model inference
    with torch.no_grad():
        outputs = model(**inputs)

    # Get logits
    logits = outputs.logits

    # Convert logits into probabilities
    probabilities = torch.nn.functional.softmax(
        logits,
        dim=-1
    )

    # Find predicted class
    predicted_class_id = logits.argmax(
        dim=-1
    ).item()

    # Confidence
    confidence = probabilities[
        0,
        predicted_class_id
    ].item()

    # Get label from model configuration
    raw_label = model.config.id2label.get(
        predicted_class_id,
        f"LABEL_{predicted_class_id}"
    )

    raw_label = str(raw_label).upper()

    # Support common binary sentiment labels
    if raw_label in [
        "POSITIVE",
        "LABEL_1",
        "1"
    ]:
        label_display = "Positive"
    else:
        label_display = "Negative"

    return label_display, confidence


st.sidebar.markdown("### ⚙️ Analysis Settings")

analysis_mode = st.sidebar.radio(
    "Select Input Type:",
    [
        "Upload Dataset (CSV/Excel)",
        "Analyze Single Text"
    ]
)


if analysis_mode == "Upload Dataset (CSV/Excel)":

    st.markdown("### 📁 Batch Dataset Processing")

    uploaded_file = st.file_uploader(
        "Choose a CSV or Excel file containing text data",
        type=["csv", "xlsx"]
    )

    if uploaded_file is not None:

        try:

            if uploaded_file.name.lower().endswith(".csv"):
                df = pd.read_csv(uploaded_file)

            else:
                df = pd.read_excel(uploaded_file)

        except Exception as e:

            st.error(
                f"❌ Error reading uploaded file: {e}"
            )

            st.stop()

        st.write(
            "📋 **Preview of Uploaded Data:**"
        )

        st.dataframe(
            df.head(10),
            use_container_width=True
        )

        if len(df.columns) == 0:

            st.warning(
                "The uploaded dataset does not contain any columns."
            )

            st.stop()

        text_column = st.selectbox(
            "Select the column containing the text:",
            df.columns
        )

        info_col1, info_col2 = st.columns(2)

        info_col1.metric(
            "Rows",
            len(df)
        )

        info_col2.metric(
            "Columns",
            len(df.columns)
        )

        if st.button(
            "🚀 Process Entire Dataset",
            type="primary"
        ):

            results_records = []

            pos_count = 0
            neg_count = 0

            progress_bar = st.progress(0)
            status_text = st.empty()

            total_rows = len(df)

            for idx, row in df.iterrows():

                text_item = row[text_column]

                # Skip missing values
                if pd.isna(text_item):
                    continue

                text_str = str(text_item).strip()

                # Skip empty strings
                if text_str == "":
                    continue

                try:

                    label_display, score = predict_sentiment(
                        text_str
                    )

                    if label_display == "Positive":
                        pos_count += 1
                    else:
                        neg_count += 1

                    results_records.append(
                        {
                            "Text Data": text_str,
                            "Predicted Sentiment": label_display,
                            "Confidence": score,
                            "Confidence (%)": f"{score:.2%}"
                        }
                    )

                except Exception as inference_error:

                    st.warning(
                        f"⚠️ Error processing row "
                        f"{idx + 1}: {inference_error}"
                    )

                # Update progress
                progress_bar.progress(
                    min((idx + 1) / total_rows, 1.0)
                )

                status_text.text(
                    f"Analyzing row {idx + 1} "
                    f"of {total_rows}..."
                )

            # Clear progress indicators
            status_text.empty()
            progress_bar.empty()

            if results_records:

                output_df = pd.DataFrame(
                    results_records
                )

                st.success(
                    f"📊 Completed! Successfully analyzed "
                    f"{len(results_records)} rows."
                )

                metric_cols = st.columns(3)

                metric_cols[0].metric(
                    "Total Analyzed",
                    len(results_records)
                )

                metric_cols[1].metric(
                    "Positive",
                    pos_count
                )

                metric_cols[2].metric(
                    "Negative",
                    neg_count
                )

                fig = go.Figure(
                    data=[
                        go.Bar(
                            x=[
                                "Positive",
                                "Negative"
                            ],
                            y=[
                                pos_count,
                                neg_count
                            ],
                            marker_color=[
                                "#2ecc71",
                                "#e74c3c"
                            ]
                        )
                    ]
                )

                fig.update_layout(
                    title="Dataset Sentiment Breakdown",
                    xaxis_title="Sentiment",
                    yaxis_title="Number of Records",
                    template="plotly_white",
                    height=400
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

                st.markdown(
                    "### 📋 Complete Analysis Breakdown"
                )

                st.dataframe(
                    output_df,
                    use_container_width=True,
                    hide_index=True
                )

                download_df = output_df.drop(
                    columns=["Confidence"],
                    errors="ignore"
                )

                csv_data = download_df.to_csv(
                    index=False
                ).encode("utf-8")

                st.download_button(
                    label="⬇️ Download Analysis Results",
                    data=csv_data,
                    file_name="sentiment_analysis_results.csv",
                    mime="text/csv"
                )

            else:

                st.warning(
                    "No valid rows were successfully processed."
                )


else:

    st.markdown(
        "### ✍️ Single Sentence Playground"
    )

    user_input = st.text_area(
        "Type or paste text to evaluate:",
        placeholder=(
            "Example: I really enjoyed this product!"
        ),
        height=150
    )

    if st.button(
        "🔍 Run Evaluation",
        type="primary"
    ):

        if user_input.strip() != "":

            with st.spinner(
                "Running AI inference..."
            ):

                try:

                    label_display, score = predict_sentiment(
                        user_input
                    )

                    if label_display == "Positive":

                        st.success(
                            f"### 😊 Positive Sentiment\n\n"
                            f"**Confidence:** {score:.2%}"
                        )

                    else:

                        st.error(
                            f"### 😞 Negative Sentiment\n\n"
                            f"**Confidence:** {score:.2%}"
                        )

                    st.markdown(
                        "#### Model Confidence"
                    )

                    st.progress(
                        score
                    )

                except Exception as e:

                    st.error(
                        f"❌ Inference error: {e}"
                    )

        else:

            st.warning(
                "Please enter some text before running "
                "the prediction."
            )

st.markdown("---")

st.caption(
    "Built with Python, PyTorch, Hugging Face Transformers "
    "and Streamlit."
)
