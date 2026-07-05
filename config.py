"""
Centralno mjesto za sve hiperparametre i podesavanja.
"""

N_SAMPLE       = 60_000   # broj instanci koji koristimo (iz ~500k)
MAX_TFIDF      = 10_000   # broj TF-IDF obelezja
N_SVD          = 100      # dimenzija nakon TruncatedSVD (za KNN)
EMBED_DIM      = 100      # Word2Vec dimenzija
MAX_SEQ_LEN    = 200      # max duzina sekvence za CNN
BATCH_SIZE_MLP = 256
BATCH_SIZE_CNN = 128
NUM_EPOCHS     = 10
LR             = 0.001
CV_FOLDS       = 5
RANDOM_STATE   = 42
