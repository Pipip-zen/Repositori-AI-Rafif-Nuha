# Example 4.23 Modified: Sentiment Analysis with Positive and Negative Sentences
# https://github.com/huggingface/transformers
# Install model support in this environment: pip install transformers torch
from transformers import pipeline

# Inisialisasi pipeline analisis sentimen
classifier = pipeline('sentiment-analysis')

# Kumpulan kalimat uji dengan sentimen positif dan negatif
sentences = [
    "The new features of this application are exceptionally smooth and user-friendly.",
    "I am extremely disappointed with the frequent system crashes and poor customer service.",
    "The performance and battery life on this laptop exceeded all my expectations.",
    "This was a complete waste of time and money, an absolutely terrible experience.",
]

# Melakukan inferensi dan mencetak hasil untuk setiap kalimat
for sentence in sentences:
    result = classifier(sentence)
    print(f"Text   : {sentence}")
    print(f"Result : {result}\n")