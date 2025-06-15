import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import confusion_matrix, roc_curve, auc
import seaborn as sns
import matplotlib.pyplot as plt

# Load model dan scaler
model = joblib.load("fraud_model_xgb.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Deteksi Penipuan Transaksi Digital", layout="centered")

st.title("💳 Prediksi Penipuan Transaksi Digital")
st.markdown("Upload data transaksi untuk dianalisis menggunakan model XGBoost.")

uploaded_file = st.file_uploader("Unggah file CSV", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    if 'Class' in data.columns:
        y_true = data['Class']
        data = data.drop('Class', axis=1)
    else:
        y_true = None

    # Preprocessing
    data_scaled = scaler.transform(data)
    y_pred = model.predict(data_scaled)
    y_proba = model.predict_proba(data_scaled)[:, 1]

    data['Prediksi'] = y_pred

    st.subheader("📊 Hasil Prediksi")
    st.write(data.head())

    st.subheader("🔎 Jumlah Prediksi")
    st.write(data['Prediksi'].value_counts().rename({0: "Normal", 1: "Penipuan"}))

    # Confusion matrix
    if y_true is not None:
        cm = confusion_matrix(y_true, y_pred)
        fig1, ax1 = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1)
        ax1.set_xlabel("Prediksi")
        ax1.set_ylabel("Sebenarnya")
        ax1.set_title("Confusion Matrix")
        st.pyplot(fig1)

        # ROC curve
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        auc_score = auc(fpr, tpr)
        fig2, ax2 = plt.subplots()
        ax2.plot(fpr, tpr, label=f"AUC = {auc_score:.2f}")
        ax2.plot([0, 1], [0, 1], 'k--')
        ax2.set_xlabel("False Positive Rate")
        ax2.set_ylabel("True Positive Rate")
        ax2.set_title("ROC Curve")
        ax2.legend()
        st.pyplot(fig2)