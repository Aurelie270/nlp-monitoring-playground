# app.py
import streamlit as st
import pandas as pd

from db import init_db, log_prediction, fetch_all_predictions
from models import predict_sentiment
from monitoring import compute_basic_metrics, build_charts

# Initialisation DB
init_db()

st.set_page_config(
    page_title="NLP Monitoring Playground",
    layout="wide"
)

st.sidebar.title("NLP Monitoring Playground")
page = st.sidebar.radio("Navigation", ["🔤 Prédiction", "📊 Monitoring", "📚 À propos"])

if page == "🔤 Prédiction":
    st.title("🔤 Analyse de sentiment avec un Transformer")

    text = st.text_area("Tape un texte (FR ou autre) :", height=150)

    true_label = st.selectbox(
        "Si tu veux, donne la 'vraie' note (1–5 étoiles) pour calculer les métriques :",
        options=[None, 1, 2, 3, 4, 5],
        format_func=lambda x: "Je ne sais pas / je ne mets rien" if x is None else f"{x} ★"
    )

    if st.button("Analyser"):
        if not text.strip():
            st.warning("Merci de renseigner un texte.")
        else:
            pred_label, confidence, probs = predict_sentiment(text)

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Résultat du modèle")
                st.metric("Note prédite", f"{pred_label} ★")
                st.metric("Confiance", f"{confidence*100:.1f} %")

            with col2:
                st.subheader("Confiance du modèle")
                st.metric("Score", f"{confidence*100:.1f} %")


            # Log dans la BDD
            log_prediction(
                text=text,
                true_label=true_label,
                pred_label=pred_label,
                confidence=confidence
            )
            st.success("Prédiction enregistrée dans la base pour le monitoring ✅")

elif page == "📊 Monitoring":
    st.title("📊 Monitoring du modèle")

    rows = fetch_all_predictions()
    if not rows:
        st.info("Aucune prédiction pour l’instant. Va d’abord dans l’onglet 'Prédiction'.")
    else:
        df = pd.DataFrame(rows)
        st.subheader("Journal des prédictions")
        st.dataframe(df[["timestamp", "text", "true_label", "pred_label", "confidence"]])

        metrics = compute_basic_metrics(df)
        st.subheader("Métriques globales")

        col1, col2, col3 = st.columns(3)
        col1.metric("Nombre d'exemples", metrics["n"])
        col2.metric("Accuracy (avec true_label)", f"{metrics['accuracy']*100:.1f} %" if metrics["accuracy"] is not None else "N/A")
        col3.metric("Confiance moyenne", f"{metrics['mean_conf']*100:.1f} %")

        st.subheader("Visualisations")
        build_charts(df)

elif page == "📚 À propos":
    st.title("📚 À propos du projet")
    st.markdown("""
Ce projet est un **playground de monitoring de modèle NLP** :
- Modèle : Transformer pré-entraîné (*sentiment analysis*).
- Base : SQLite.
- Interface : Streamlit.
- Monitoring : métriques + graphiques, mis à jour au fil des prédictions.

Il a été conçu comme projet d'auto-formation pour :
- MLOps (monitoring, journaux de prédiction),
- Bases de données,
- Métriques d'évaluation,
- Transformers.
""")
