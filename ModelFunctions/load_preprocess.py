import re
import string
from nltk.corpus import stopwords
import spacy
import pandas as pd

# Load SpaCy English model
nlp = spacy.load("en_core_web_sm")

# Contraction map
contractions = {
    "wouldn't": "would not", "couldn't": "could not", "shouldn't": "should not",
    "won't": "will not", "don't": "do not", "doesn't": "does not",
    "didn't": "did not", "can't": "cannot", "isn't": "is not",
    "aren't": "are not", "wasn't": "was not", "weren't": "were not",
    "hasn't": "has not", "haven't": "have not", "hadn't": "had not",
    "it's": "it is", "I'm": "I am", "you're": "you are", "they're": "they are"
}

# Remove punctuations
def remove_punctuations(text):
    return text.translate(str.maketrans('', '', string.punctuation))

# Lemmatize text
def lemmatize_text(text):
    doc = nlp(text)
    return ' '.join([token.lemma_ for token in doc if not token.is_punct])

# Preprocessing function
def preprocess_text(text, stopwords_remove=False, lemmatize=False):
    # Replace curly quotes
    text = text.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    # Lowercase
    text = text.lower()
    # Remove newlines and normalize the whitespace
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    # Replace contradictions
    for k, v in contractions.items():
        text = re.sub(rf"\b{k}\b", v, text, flags=re.IGNORECASE)
    # Remove punctuations
    text = remove_punctuations(text)
    # Stopwords remove (optional)
    if stopwords_remove:
        stop_words = set(stopwords.words('english'))
        text = ' '.join(w for w in text.split() if w not in stop_words)
    # Lemmatize text (optional)
    if lemmatize:
        text = lemmatize_text(text)

    return text


def load_and_preprocess(stopwords_remove=False, lemmatize=False):
    # Load dataset
    df = pd.read_csv("../Data/Sample_Fellowship_Dataset.csv")
    # We preprocess without stopwords removing and lemmatization they dont help here
    df['Chunk'] = df['Chunk'].astype(str).apply(lambda x: preprocess_text(x, stopwords_remove=stopwords_remove, lemmatize=lemmatize))
    chunks = df['Chunk'].tolist()
    labels = df['Fellowship'].tolist()

    return chunks, labels

