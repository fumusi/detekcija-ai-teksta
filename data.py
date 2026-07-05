"""
Ucitavanje, ciscenje i uzorkovanje dataseta.
"""

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import N_SAMPLE, RANDOM_STATE


def load_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f'Ukupno instanci: {len(df)}')
    print(f'Kolone: {list(df.columns)}')
    print()
    print('Raspodjela klasa:')
    print(df['generated'].value_counts())

    df = df.dropna(subset=['text'])
    df['text'] = df['text'].astype(str)
    return df


def sample_balanced(df: pd.DataFrame, n: int = N_SAMPLE) -> pd.DataFrame:
    """Stratifikovano uzorkovanje — jednako instanci po klasi."""
    df_sample = (
        df.groupby('generated', group_keys=False)
          .apply(lambda x: x.sample(min(len(x), n // 2), random_state=RANDOM_STATE))
          .reset_index(drop=True)
    )
    print(f'\nUzorkovano: {len(df_sample)} instanci')
    print(df_sample['generated'].value_counts())
    return df_sample


def preprocess_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def plot_class_distribution(df: pd.DataFrame, save_path: str = 'plots/01_raspodjela_klasa.png'):
    plt.figure(figsize=(5, 4))
    df['generated'].value_counts().sort_index().plot(
        kind='bar', color=['steelblue', 'salmon']
    )
    plt.xticks([0, 1], ['Covjek (0)', 'AI (1)'], rotation=0)
    plt.title('Raspodjela klasa')
    plt.ylabel('Broj instanci')
    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.close()
    print(f'Sacuvano: {save_path}')
