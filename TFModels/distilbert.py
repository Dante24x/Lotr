import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, train_test_split
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from itertools import product
import random
import matplotlib.pyplot as plt
import seaborn as sns
from torch.optim import AdamW

# 1. Set random seed
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# 2. Load and encode dataset
df = pd.read_csv("../Data/Sample_Fellowship_Dataset.csv")
texts = df["Chunk"].astype(str).tolist()
labels = df["Fellowship"].tolist()

label_encoder = LabelEncoder()
y_all = label_encoder.fit_transform(labels)
num_classes = len(label_encoder.classes_)

# 3. Tokenize text using DistilBERT
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
tokenized = tokenizer(texts, padding=True, truncation=True, max_length=256, return_tensors="pt")

# 4. Dataset class for DistilBERT
class DistilBERTDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

# 5. Training and validation for one fold
def train_model(model, train_loader, val_loader, device, lr, epochs):
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr)
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            preds = torch.argmax(outputs.logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch["labels"].cpu().numpy())

    return f1_score(all_labels, all_preds, average="macro")

# 6. Final training on full training set
def train_final_model(model, train_loader, device, lr, epochs):
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr)
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

# 7. DistilBERT Grid search with cross-validation and final test
def run_distilbert_kfold_grid_search():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    grid = {
        "lr": [2e-5, 3e-5],
        "batch_size": [16, 32],
        "epochs": [3, 4]
    }

    # Split off clean test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        texts, y_all, test_size=0.2, stratify=y_all, random_state=42
    )

    best_f1 = 0.0
    best_params = None
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for lr, batch_size, epochs in product(*grid.values()):
        set_seed(42)
        fold_scores = []

        for train_idx, val_idx in skf.split(X_temp, y_temp):
            X_train = [X_temp[i] for i in train_idx]
            X_val = [X_temp[i] for i in val_idx]
            y_train = [y_temp[i] for i in train_idx]
            y_val = [y_temp[i] for i in val_idx]

            enc_train = tokenizer(X_train, padding=True, truncation=True, max_length=256, return_tensors="pt")
            enc_val = tokenizer(X_val, padding=True, truncation=True, max_length=256, return_tensors="pt")

            train_loader = DataLoader(DistilBERTDataset(enc_train, y_train), batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(DistilBERTDataset(enc_val, y_val), batch_size=batch_size)

            model = DistilBertForSequenceClassification.from_pretrained(
                "distilbert-base-uncased", num_labels=num_classes
            )

            f1 = train_model(model, train_loader, val_loader, device, lr, epochs)
            fold_scores.append(f1)

        avg_f1 = np.mean(fold_scores)
        print(f"CV-F1={avg_f1:.4f} | lr={lr}, batch_size={batch_size}, epochs={epochs}")

        if avg_f1 > best_f1:
            best_f1 = avg_f1
            best_params = (lr, batch_size, epochs)

    print(f"\nBest CV F1={best_f1:.4f} with params: {best_params}")

    # Final evaluation on held-out test set
    print("\nFinal Evaluation on Test Set...")
    enc_train = tokenizer(X_temp, padding=True, truncation=True, max_length=256, return_tensors="pt")
    enc_test = tokenizer(X_test, padding=True, truncation=True, max_length=256, return_tensors="pt")

    train_loader = DataLoader(DistilBERTDataset(enc_train, y_temp), batch_size=best_params[1], shuffle=True)
    test_loader = DataLoader(DistilBERTDataset(enc_test, y_test), batch_size=best_params[1])

    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=num_classes
    )

    train_final_model(model, train_loader, device, lr=best_params[0], epochs=best_params[2])

    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            preds = torch.argmax(outputs.logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch["labels"].cpu().numpy())

    test_f1 = f1_score(all_labels, all_preds, average="macro")
    print(f"\nFinal Test F1: {test_f1:.4f}")
    print("\nClassification Report: DistilBERT (Test Set)")
    print(classification_report(all_labels, all_preds, target_names=label_encoder.classes_))

    # Plot confusion matrix
    conf_matrix = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6, 4))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Grays",
                xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.title("Confusion Matrix: DistilBERT (Test Set)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig("confusion_matrix_distilbert.png")
    plt.show()

run_distilbert_kfold_grid_search()
