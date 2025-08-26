import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title=" Fraud Detection", layout="wide")
st.title("🔍 Fraud Detection App")
st.markdown("Upload your transaction data CSV to predict fraudulent activities.")

@st.cache_resource
def load_model():
    return joblib.load("model/xgb_model.pkl")

model = load_model()


def predict(df):
    prediction = model.predict(df)
    proba = model.predict_proba(df)[:, 1]
    return prediction, proba

uploaded_file = st.file_uploader("📁 Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("📊 Uploaded Data Preview")
    st.dataframe(df.head())

    required_columns = model.feature_names_in_
    if not all(col in df.columns for col in required_columns):
        st.error(f"Uploaded file must contain the following columns:\n\n{list(required_columns)}")
    else:
        X_input = df[required_columns]
        preds, probs = predict(X_input)

        result_df = df.copy()
        result_df["Fraud_Prediction"] = preds
        result_df["Fraud_Probability"] = np.round(probs, 4)

        st.subheader("🔎 Prediction Results")
        st.dataframe(result_df.sort_values("Fraud_Probability", ascending=False).reset_index(drop=True))

        st.subheader("📈 Fraud vs Normal Distribution")
        st.bar_chart(result_df["Fraud_Prediction"].value_counts())

        csv = result_df.to_csv(index=False).encode('utf-8')
        st.download_button("💾 Download Results as CSV", data=csv, file_name='fraud_detection_results.csv', mime='text/csv')
