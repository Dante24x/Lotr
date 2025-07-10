import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import StratifiedKFold, train_test_split
from nltk.tokenize import word_tokenize
from itertools import product
from ModelFunctions import load_preprocess
import random
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# 1. Set random seed and deterministic behavior
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# 2. Load and preprocess
df = pd.read_csv("../Data/Sample_Fellowship_Dataset.csv")
texts = df['Chunk'].astype(str).apply(lambda x: load_preprocess.preprocess_text(x, stopwords_remove=True, lemmatize=True)).tolist()
labels = df['Fellowship'].tolist()

label_encoder = LabelEncoder()
y_all = label_encoder.fit_transform(labels)
num_classes = len(label_encoder.classes_)

# 3. Tokenize and vocab
tokenized_texts = [word_tokenize(text.lower()) for text in texts]
vocab = {'<PAD>': 0, '<UNK>': 1}
for tokens in tokenized_texts:
    for token in tokens:
        if token not in vocab:
            vocab[token] = len(vocab)

sequences = [[vocab.get(token, vocab['<UNK>']) for token in tokens] for tokens in tokenized_texts]
max_len = 150
padded_sequences = [seq[:max_len] + [0]*(max_len - len(seq)) if len(seq) < max_len else seq[:max_len] for seq in sequences]

# 4. Dataset Class
class FellowshipDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.LongTensor(X)
        self.y = torch.LongTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# 4. Load GloVe embeddings
def load_glove(path='../Data/glove.6B.300d.txt'):
    embedding_dict = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.strip().split()
            word = values[0]
            vector = np.array(values[1:], dtype='float32')
            embedding_dict[word] = vector
    return embedding_dict

# 6. Build embedding matrix for GloVe
def build_embedding_matrix(vocab, embedding_dict, dim=300):
    matrix = np.zeros((len(vocab), dim))
    for word, idx in vocab.items():
        vector = embedding_dict.get(word)
        if vector is not None:
            matrix[idx] = vector
        else:
            matrix[idx] = np.random.normal(scale=0.6, size=(dim,))
    return torch.FloatTensor(matrix)

# 7. GRU Bi attentional Model
class GRUModel(nn.Module):
    def __init__(self, embedding_matrix, hidden_dim, num_layers, dropout, num_classes):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=False)
        self.gru = nn.GRU(
            input_size=embedding_matrix.shape[1],
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True
        )
        self.attn_hidden = nn.Linear(hidden_dim * 2, hidden_dim)
        self.context_vector = nn.Parameter(torch.rand(hidden_dim))
        self.fc1 = nn.Linear(hidden_dim * 2, 128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        gru_out, _ = self.gru(x)
        score = torch.tanh(self.attn_hidden(gru_out))
        score = torch.matmul(score, self.context_vector)
        attn_weights = torch.softmax(score, dim=1).unsqueeze(2)
        context = torch.sum(attn_weights * gru_out, dim=1)
        out = self.relu(self.fc1(context))
        out = self.dropout(out)
        return self.fc2(out)

# 8. Training and validating function for cross validation in grid search
def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=5):
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    return f1_score(all_labels, all_preds, average='macro')

# 9. Train final model with all the train dataset
def train_final_model(model, train_loader, criterion, optimizer, device, num_epochs=5):
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

# 10. CNN Grid Search with Clean K-Fold CV + Final Test Evaluation
def run_gru_kfold_grid_search():
    glove = load_glove()
    emb_matrix = build_embedding_matrix(vocab, glove)

    # Use CUDA so we can use our GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Hyperparameter combination grid for our grid search
    grid = {
        'hidden_dim': [64, 96, 128],
        'num_layers': [2, 3],
        'dropout': [0.2, 0.3, 0.4, 0.5],
        'lr': [1e-3, 5e-4, 3e-4],
        'batch_size': [32],
        'num_epochs': [10]
    }

    # Hold-out test set (unseen untill the end)
    X_temp, X_test, y_temp, y_test = train_test_split(padded_sequences, y_all, test_size=0.2, stratify=y_all, random_state=42)
    temp_dataset = FellowshipDataset(X_temp, y_temp)
    test_dataset = FellowshipDataset(X_test, y_test)

    # Best performance values init and StratifiedKFold init
    best_f1 = 0.0
    best_params = None
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Grid search loop
    for hidden_dim, num_layers, dropout, lr, batch_size, num_epochs in product(*grid.values()):
        set_seed(42)
        fold_scores = []

        for train_idx, val_idx in skf.split(X_temp, y_temp):
            # Train and val sets for each split
            train_subset = torch.utils.data.Subset(temp_dataset, train_idx)
            val_subset = torch.utils.data.Subset(temp_dataset, val_idx)

            train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_subset, batch_size=batch_size)

            # New def of model each time both for splits and grid combinations
            model = GRUModel(
                embedding_matrix=emb_matrix,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                dropout=dropout,
                num_classes=num_classes
            )

            # Adam optimizer and criterion based on CrossEntropyLoss
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            criterion = nn.CrossEntropyLoss()

            # Train model for each fold and keep the avg of them
            f1 = train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=num_epochs)
            fold_scores.append(f1)

        # Print avg for the fold set of this grid search combination
        avg_f1 = np.mean(fold_scores)
        print(f"CV-F1={avg_f1:.4f} | hidden={hidden_dim}, layers={num_layers}, dropout={dropout}, lr={lr}, batch_size={batch_size}, epochs={num_epochs}")

        # Keep the best hyperparameter combination at the end, based on avg of all folds F1
        if avg_f1 > best_f1:
            best_f1 = avg_f1
            best_params = (hidden_dim, num_layers, dropout, lr, batch_size, num_epochs)

    # Print the best average score of all grid search
    print(f"\n Best CV F1={best_f1:.4f} with params: {best_params}")

    # 11. Final Evaluation on clean test set
    print("\n Final Evaluation on Held-out Test Set...")
    final_train_loader = DataLoader(temp_dataset, batch_size=best_params[4], shuffle=True)
    final_test_loader = DataLoader(test_dataset, batch_size=best_params[4])

    # New def of model
    final_model = GRUModel(
        embedding_matrix=emb_matrix,
        hidden_dim=best_params[0],
        num_layers=best_params[1],
        dropout=best_params[2],
        num_classes=num_classes
    )

    final_model.to(device)

    # Optimizer with best combination's learning rate
    optimizer = torch.optim.Adam(final_model.parameters(), lr=best_params[3])
    criterion = nn.CrossEntropyLoss()

    # Train the model with all the train dataset without validation split
    train_final_model(final_model, final_train_loader, criterion, optimizer, device, num_epochs=best_params[5])

    # Take final test datasets results
    final_model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in final_test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = final_model(inputs)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    test_f1 = f1_score(all_labels, all_preds, average='macro')
    # Print final tests f1 score
    print(f"\n Final Test F1 on Held-out Set: {test_f1:.4f}")

    # Print classification report
    print("\n Classification Report: GRU Model (Test Set)")
    print(classification_report(all_labels, all_preds, target_names=label_encoder.classes_))

    # Plot and save confusion matrix
    conf_matrix = confusion_matrix(all_labels, all_preds)
    label_names = label_encoder.classes_
    color = "Greens"

    plt.figure(figsize=(6, 4))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap=color,
                xticklabels=label_names, yticklabels=label_names)
    plt.title("Confusion Matrix: GRU Model (Test Set)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig("confusion_matrix_gru.png")
    plt.show()

run_gru_kfold_grid_search()
