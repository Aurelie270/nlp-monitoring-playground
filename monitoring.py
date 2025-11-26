# monitoring.py
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

def compute_basic_metrics(df: pd.DataFrame):
    n = len(df)
    acc = None
    if "true_label" in df.columns and df["true_label"].notna().any():
        valid = df.dropna(subset=["true_label"])
        if len(valid) > 0:
            acc = (valid["true_label"] == valid["pred_label"]).mean()
    mean_conf = df["confidence"].mean() if "confidence" in df.columns else None

    return {
        "n": n,
        "accuracy": acc,
        "mean_conf": mean_conf
    }

def build_charts(df: pd.DataFrame):
    # Répartition des prédictions
    st.markdown("### Répartition des notes prédites")
    fig1, ax1 = plt.subplots()
    df["pred_label"].value_counts().sort_index().plot(kind="bar", ax=ax1)
    ax1.set_xlabel("Note prédite (★)")
    ax1.set_ylabel("Nombre")
    st.pyplot(fig1)

    # Confiance dans le temps
    st.markdown("### Confiance dans le temps")
    df_sorted = df.sort_values("timestamp")
    fig2, ax2 = plt.subplots()
    ax2.plot(df_sorted["timestamp"], df_sorted["confidence"], marker="o")
    ax2.set_xticklabels(df_sorted["timestamp"], rotation=45, ha="right")
    ax2.set_ylabel("Confiance")
    st.pyplot(fig2)
