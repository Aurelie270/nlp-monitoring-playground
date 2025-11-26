from transformers import pipeline

classifier = pipeline(
    "sentiment-analysis",
    model="nlptown/bert-base-multilingual-uncased-sentiment"
)

def predict_sentiment(text: str):
    result = classifier(text)[0]  # {'label': '4 stars', 'score': 0.89}

    label = result["label"]
    stars = int(label.split()[0])
    confidence = float(result["score"])

    return stars, confidence, None
