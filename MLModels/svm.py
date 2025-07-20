import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
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

# Grid Search setup
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
    'gamma': ['scale', 'auto'],
    'class_weight': [None, 'balanced']
}

grid_search = GridSearchCV(
    SVC(),
    param_grid,
    cv=5,
    scoring='f1_macro',
    n_jobs=-1,
    verbose=2
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
print("\n Classification Report: SVM Model (Test Set)")
print(classification_report(y_test, y_test_pred))

# Confusion matrix
labels = ["Negative", "Neutral", "Positive"]
conf_matrix = confusion_matrix(y_test, y_test_pred, labels=labels)
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap="Blues",
            xticklabels=labels, yticklabels=labels)
plt.title("Confusion Matrix: SVM Model (Test Set)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig("confusion_matrix_svm.png")
plt.show()
