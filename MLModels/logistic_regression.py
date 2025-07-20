import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from ModelFunctions import load_preprocess

# Load and preprocess data
df = pd.read_csv("../Data/Sample_Fellowship_Dataset.csv")
df['Chunk'] = df['Chunk'].astype(str).apply(load_preprocess.preprocess_text)
X = df['Chunk']
y = df['Fellowship']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# TF-IDF vectorization
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Grid Search for Logistic Regression
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'penalty': ['l2'],  # 'l1' requires solver='liblinear'
    'class_weight': [None, 'balanced'],
    'solver': ['lbfgs']  # use lbfgs with l2
}

grid_search = GridSearchCV(
    LogisticRegression(max_iter=1000),
    param_grid,
    cv=5,
    scoring='f1_macro',
    verbose=2,
    n_jobs=-1
)

# Run grid search
grid_search.fit(X_train_tfidf, y_train)

# Best model summary
print("\n Best Parameters Found:")
print(grid_search.best_params_)
print("\n Best Cross-Validation F1 (macro):")
print(grid_search.best_score_)

# Best model
best_model = grid_search.best_estimator_

# Predict on test set
y_test_pred = best_model.predict(X_test_tfidf)
print("\n Classification Report: Logistic Regression (Test Set)")
print(classification_report(y_test, y_test_pred))

# Confusion matrix
labels = ["Negative", "Neutral", "Positive"]
conf_matrix = confusion_matrix(y_test, y_test_pred, labels=labels)
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap="Reds",
            xticklabels=labels, yticklabels=labels)
plt.title("Confusion Matrix: Logistic Regression (Test Set)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig("confusion_matrix_logreg.png")
plt.show()
