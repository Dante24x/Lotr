import pandas as pd
from sklearn.ensemble import RandomForestClassifier
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

# Random Forest Grid Search setup
param_grid = {
    'n_estimators': [100, 200, 500],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'class_weight': [None, 'balanced']
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
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
print("\n📋 Classification Report: Random Forest Model (Test Set):")
print(classification_report(y_test, y_test_pred))

# Confusion matrix
labels = ["Positive", "Negative", "Neutral"]
conf_matrix = confusion_matrix(y_test, y_test_pred, labels=labels)
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap="Greens",
            xticklabels=labels, yticklabels=labels)
plt.title("Confusion Matrix: Random Forest Model (Test Set)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig("confusion_matrix_random_forest.png")
plt.show()
