"""
Detekcija AI-generisanog teksta
Dimitrije Ignjic RA129/2022
Osnove racunarske inteligencije, 2025/2026

Pokretanje:
    python main.py

Dataset:
    https://www.kaggle.com/datasets/shanegerami/ai-vs-human-text/data
    Fajl AI_Human.csv smjestiti u isti folder kao main.py
"""

import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

from config import (
    N_SAMPLE, MAX_TFIDF, N_SVD, EMBED_DIM, MAX_SEQ_LEN,
    BATCH_SIZE_MLP, BATCH_SIZE_CNN, NUM_EPOCHS, LR, RANDOM_STATE
)
from data import load_dataset, sample_balanced, preprocess_text, plot_class_distribution
from features import (
    build_numerical_matrix, plot_numerical_features,
    build_tfidf, train_word2vec, build_word2vec_assets, texts_to_sequences
)
from models import MLP, TextCNN
from train import train_model, predict, make_loader, plot_loss
from evaluate import (
    print_and_plot_metrics, plot_comparison,
    kfold_knn, kfold_torch
)

# ─────────────────────────────────────────
os.makedirs('plots', exist_ok=True)

np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Uredjaj: {device}\n')


# ═══════════════════════════════════════════════════════════════
# 1. UCITAVANJE I UZORKOVANJE
# ═══════════════════════════════════════════════════════════════

print('=' * 60)
print('1. UCITAVANJE PODATAKA')
print('=' * 60)

df = load_dataset('AI_Human.csv')
df_sample = sample_balanced(df, N_SAMPLE)
plot_class_distribution(df_sample, 'plots/01_raspodjela_klasa.png')


# ═══════════════════════════════════════════════════════════════
# 2. EKSTRAKCIJA OBELEZJA
# ═══════════════════════════════════════════════════════════════

print('\n' + '=' * 60)
print('2. EKSTRAKCIJA OBELEZJA')
print('=' * 60)

labels = df_sample['generated'].values.astype(int)

# Numericka obelezja
X_num = build_numerical_matrix(df_sample['text'])
plot_numerical_features(X_num, labels, 'plots/02_numericka_obelezja.png')

# Pretprocesiranje
print('\nPretprocesiranje teksta...')
texts_clean = df_sample['text'].apply(preprocess_text).tolist()

# TF-IDF
X_tfidf, tfidf_vectorizer = build_tfidf(texts_clean)

# Word2Vec
tokenized = [t.split() for t in texts_clean]
w2v = train_word2vec(tokenized)
word2idx, embedding_matrix = build_word2vec_assets(w2v)
X_sequences = texts_to_sequences(tokenized, word2idx)


# ═══════════════════════════════════════════════════════════════
# 3. PODJELA TRENING / TEST (80/20)
# ═══════════════════════════════════════════════════════════════

print('\n' + '=' * 60)
print('3. PODJELA TRENING/TEST (80/20)')
print('=' * 60)

idx = np.arange(len(labels))
idx_train, idx_test = train_test_split(idx, test_size=0.2,
                                       random_state=RANDOM_STATE, stratify=labels)

X_num_train,  X_num_test  = X_num[idx_train],        X_num[idx_test]
X_tfidf_train, X_tfidf_test = X_tfidf[idx_train],    X_tfidf[idx_test]
X_seq_train,  X_seq_test   = X_sequences[idx_train], X_sequences[idx_test]
y_train, y_test             = labels[idx_train],      labels[idx_test]

print(f'Trening: {len(idx_train)}, Test: {len(idx_test)}')
print(f'Klase u treningu — 0: {(y_train==0).sum()}, 1: {(y_train==1).sum()}')

results = {}


# ═══════════════════════════════════════════════════════════════
# 4. MODEL 1 — KNN
# ═══════════════════════════════════════════════════════════════

print('\n' + '=' * 60)
print('4. MODEL 1: KNN')
print('=' * 60)

# TF-IDF -> TruncatedSVD + numericka obelezja
print(f'TruncatedSVD: {X_tfidf.shape[1]} -> {N_SVD} dimenzija...')
svd = TruncatedSVD(n_components=N_SVD, random_state=RANDOM_STATE)
X_svd_train = svd.fit_transform(X_tfidf_train)
X_svd_test  = svd.transform(X_tfidf_test)

scaler_num = StandardScaler()
X_num_train_s = scaler_num.fit_transform(X_num_train)
X_num_test_s  = scaler_num.transform(X_num_test)

X_knn_train = np.hstack([X_svd_train, X_num_train_s])
X_knn_test  = np.hstack([X_svd_test,  X_num_test_s])
print(f'KNN ulaz — trening: {X_knn_train.shape}, test: {X_knn_test.shape}')

# Odabir optimalnog K
k_values = [3, 5, 7, 11, 15]
k_scores = []
small_idx = np.random.choice(len(X_knn_train), size=10_000, replace=False)
X_small, y_small = X_knn_train[small_idx], y_train[small_idx]

print('\nOdabir optimalnog K (3-fold CV na 10 000 instanci):')
for k in k_values:
    score = cross_val_score(
        KNeighborsClassifier(n_neighbors=k, metric='euclidean', n_jobs=-1),
        X_small, y_small, cv=3, scoring='f1_weighted'
    ).mean()
    k_scores.append(score)
    print(f'  K={k:2d} -> F1={score:.4f}')

best_k = k_values[int(np.argmax(k_scores))]
print(f'Najbolji K: {best_k}')

import matplotlib.pyplot as plt
plt.figure(figsize=(6, 4))
plt.plot(k_values, k_scores, '-o', color='steelblue')
plt.xlabel('K')
plt.ylabel('F1-score (3-fold CV)')
plt.title('KNN — Odabir optimalnog K')
plt.grid(True)
plt.tight_layout()
plt.savefig('plots/03_knn_odabir_k.png', dpi=100)
plt.close()

print(f'\nTreniranje KNN (K={best_k})...')
knn = KNeighborsClassifier(n_neighbors=best_k, metric='euclidean', n_jobs=-1)
knn.fit(X_knn_train, y_train)
y_pred_knn = knn.predict(X_knn_test)
results['KNN'] = print_and_plot_metrics(y_test, y_pred_knn, 'KNN', 'plots/cm_knn.png')


# ═══════════════════════════════════════════════════════════════
# 5. MODEL 2 — MLP
# ═══════════════════════════════════════════════════════════════

print('\n' + '=' * 60)
print('5. MODEL 2: MLP (Viseslojna neuronska mreza)')
print('=' * 60)

scaler_tfidf = StandardScaler(with_mean=False)
X_mlp_train_s = scaler_tfidf.fit_transform(X_tfidf_train)
X_mlp_test_s  = scaler_tfidf.transform(X_tfidf_test)

X_mlp_train_t = torch.FloatTensor(X_mlp_train_s.toarray())
X_mlp_test_t  = torch.FloatTensor(X_mlp_test_s.toarray())
y_train_t     = torch.LongTensor(y_train)
y_test_t      = torch.LongTensor(y_test)

mlp = MLP(input_size=MAX_TFIDF).to(device)
mlp_loader = make_loader(X_mlp_train_t, y_train_t, BATCH_SIZE_MLP)

print('Treniranje MLP...')
mlp_hist = train_model(mlp, mlp_loader, nn.CrossEntropyLoss(),
                       optim.Adam(mlp.parameters(), lr=LR), NUM_EPOCHS, device)
plot_loss(mlp_hist, 'MLP — Kretanje loss funkcije', 'plots/04_mlp_loss.png', 'steelblue')

y_pred_mlp = predict(mlp, X_mlp_test_t, device)
results['MLP'] = print_and_plot_metrics(y_test, y_pred_mlp, 'MLP', 'plots/cm_mlp.png')


# ═══════════════════════════════════════════════════════════════
# 6. MODEL 3 — CNN 1D
# ═══════════════════════════════════════════════════════════════

print('\n' + '=' * 60)
print('6. MODEL 3: CNN (1D, Word2Vec embeddings)')
print('=' * 60)

VOCAB_SIZE = len(word2idx) + 1

X_seq_train_t = torch.LongTensor(X_seq_train)
X_seq_test_t  = torch.LongTensor(X_seq_test)

cnn = TextCNN(VOCAB_SIZE, EMBED_DIM, embedding_matrix).to(device)
print(cnn)

cnn_loader = make_loader(X_seq_train_t, y_train_t, BATCH_SIZE_CNN)

print('\nTreniranje CNN...')
cnn_hist = train_model(cnn, cnn_loader, nn.CrossEntropyLoss(),
                       optim.Adam(cnn.parameters(), lr=LR), NUM_EPOCHS, device)
plot_loss(cnn_hist, 'CNN — Kretanje loss funkcije', 'plots/05_cnn_loss.png', 'salmon')

y_pred_cnn = predict(cnn, X_seq_test_t, device)
results['CNN'] = print_and_plot_metrics(y_test, y_pred_cnn, 'CNN (1D, Word2Vec)', 'plots/cm_cnn.png')


# ═══════════════════════════════════════════════════════════════
# 7. K-FOLD UNAKRSNA VALIDACIJA (k=5)
# ═══════════════════════════════════════════════════════════════

print('\n' + '=' * 60)
print('7. K-FOLD UNAKRSNA VALIDACIJA (k=5)')
print('=' * 60)

kfold_knn(X_knn_train, y_train, best_k)

kfold_torch(
    model_fn=lambda: MLP(input_size=MAX_TFIDF),
    X_tensor=X_mlp_train_t, y_tensor=y_train_t,
    device=device, batch_size=BATCH_SIZE_MLP,
    model_name='MLP', n_sample=15_000
)

kfold_torch(
    model_fn=lambda: TextCNN(VOCAB_SIZE, EMBED_DIM, embedding_matrix),
    X_tensor=X_seq_train_t, y_tensor=y_train_t,
    device=device, batch_size=BATCH_SIZE_CNN,
    model_name='CNN', n_sample=10_000
)


# ═══════════════════════════════════════════════════════════════
# 8. POREĐENJE MODELA
# ═══════════════════════════════════════════════════════════════

print('\n' + '=' * 60)
print('8. POREĐENJE MODELA')
print('=' * 60)

plot_comparison(results, save_dir='plots')

print('\n' + '=' * 60)
print('GOTOVO — svi grafici su u folderu plots/')
print('=' * 60)
