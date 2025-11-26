NLP Monitoring Playground

Petit projet d’apprentissage pour comprendre le fonctionnement du monitoring d’un modèle NLP.
L’application utilise Streamlit, un modèle Transformers pour l’analyse de sentiment, et une base SQLite pour enregistrer les prédictions et afficher des métriques simples.

Fonctionnalités
- Analyse de sentiment avec un modèle pré-entraîné (1★ à 5★)
- Enregistrement automatique des prédictions dans une base locale
- Tableau de bord de monitoring avec :
  - journal des prédictions
  - accuracy (si un label vrai est donné)
  - confiance moyenne
  - graphiques (répartition des notes, confiance dans le temps)

Technologies
- Python
- Streamlit
- HuggingFace Transformers
- SQLite
- Pandas / Matplotlib

