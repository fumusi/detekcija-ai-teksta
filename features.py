"""
Ekstrakcija obelezja:
  - Numericka obelezja (TTR, duzine, interpunkcija...)
  - TF-IDF vektorizacija
  - Word2Vec embeddings
"""

import re
import string

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec

from config import MAX_TFIDF, EMBED_DIM, MAX_SEQ_LEN, RANDOM_STATE


# ─────────────────────────────────────────
# Numericka obelezja
# ─────────────────────────────────────────

def extract_numerical_features(text: str) -> dict:
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    words = re.findall(r'\b\w+\b', text.lower())

    if not words:
        return {k: 0.0 for k in [
            'avg_sentence_len', 'sentence_len_std', 'avg_word_len',
            'ttr', 'punct_freq', 'num_sentences'
        ]}

    sent_lens = [len(re.findall(r'\b\w+\b', s)) for s in sentences]

    return {
        'avg_sentence_len': float(np.mean(sent_lens)) if sent_lens else 0.0,
        'sentence_len_std': float(np.std(sent_lens)) if len(sent_lens) > 1 else 0.0,
        'avg_word_len':     float(np.mean([len(w) for w in words])),
        'ttr':              len(set(words)) / len(words),
        'punct_freq':       sum(1 for c in text if c in string.punctuation) / max(len(text), 1),
        'num_sentences':    float(len(sentences))
    }


def build_numerical_matrix(texts: pd.Series) -> np.ndarray:
    print('Ekstrakcija numerickih obelezja...')
    feats = texts.apply(extract_numerical_features)
    df_num = pd.DataFrame(list(feats))
    print(df_num.describe().round(4))
    return df_num.values


def plot_numerical_features(df_num: np.ndarray, labels: np.ndarray,
                            save_path: str = 'plots/02_numericka_obelezja.png'):
    cols = ['avg_sentence_len', 'sentence_len_std', 'avg_word_len', 'ttr', 'punct_freq']
    df_plot = pd.DataFrame(df_num, columns=[
        'avg_sentence_len', 'sentence_len_std', 'avg_word_len',
        'ttr', 'punct_freq', 'num_sentences'
    ])
    df_plot['generated'] = labels

    fig, axes = plt.subplots(1, len(cols), figsize=(18, 4))
    for ax, feat in zip(axes, cols):
        for cls, label, color in [(0, 'Covjek', 'steelblue'), (1, 'AI', 'salmon')]:
            ax.hist(df_plot[df_plot['generated'] == cls][feat],
                    bins=40, alpha=0.6, label=label, color=color, density=True)
        ax.set_title(feat, fontsize=9)
        ax.legend(fontsize=7)
    plt.suptitle('Distribucija numerickih obelezja po klasama', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f'Sacuvano: {save_path}')


# ─────────────────────────────────────────
# TF-IDF
# ─────────────────────────────────────────

def build_tfidf(texts: list) -> tuple:
    """Vraca (X_tfidf matrica, fitted TfidfVectorizer)."""
    print(f'TF-IDF vektorizacija (max {MAX_TFIDF} obelezja)...')
    tfidf = TfidfVectorizer(max_features=MAX_TFIDF, sublinear_tf=True)
    X = tfidf.fit_transform(texts)
    print(f'TF-IDF matrica: {X.shape}')
    return X, tfidf


# ─────────────────────────────────────────
# Word2Vec
# ─────────────────────────────────────────

def train_word2vec(tokenized: list) -> Word2Vec:
    print(f'Treniranje Word2Vec (dim={EMBED_DIM})...')
    model = Word2Vec(
        sentences=tokenized,
        vector_size=EMBED_DIM,
        window=5,
        min_count=5,
        workers=4,
        epochs=5,
        seed=RANDOM_STATE
    )
    print(f'Velicina rjecnika: {len(model.wv)}')
    return model


def build_word2vec_assets(w2v: Word2Vec) -> tuple:
    """Vraca (word2idx rjecnik, embedding_matrix numpy array)."""
    word2idx = {w: i + 1 for i, w in enumerate(w2v.wv.index_to_key)}

    embedding_matrix = np.zeros((len(word2idx) + 1, EMBED_DIM))
    for word, idx in word2idx.items():
        embedding_matrix[idx] = w2v.wv[word]

    return word2idx, embedding_matrix


def texts_to_sequences(tokenized: list, word2idx: dict) -> np.ndarray:
    print('Konverzija tekstova u sekvence...')

    def _encode(tokens):
        seq = [word2idx.get(w, 0) for w in tokens]
        if len(seq) < MAX_SEQ_LEN:
            seq += [0] * (MAX_SEQ_LEN - len(seq))
        else:
            seq = seq[:MAX_SEQ_LEN]
        return seq

    X = np.array([_encode(t) for t in tokenized])
    print(f'Matrica sekvenci: {X.shape}')
    return X
