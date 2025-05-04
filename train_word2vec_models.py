from gensim.models import Word2Vec
import pandas as pd

# Veriyi yükle (daha önce oluşturulan veri)
df = pd.read_csv("e-ticaret_urun_yorumlari_cleaned.csv")  # Lemmatized ve stemmed sütunları içermeli
df['metin_lemmatized'] = df['metin_lemmatized'].apply(eval)
df['metin_stemmed'] = df['metin_stemmed'].apply(eval)

# Parametre kombinasyonları
parameters = [
    {'model_type': 'cbow', 'window': 2, 'vector_size': 100},
    {'model_type': 'skipgram', 'window': 2, 'vector_size': 100},
    {'model_type': 'cbow', 'window': 4, 'vector_size': 100},
    {'model_type': 'skipgram', 'window': 4, 'vector_size': 100},
    {'model_type': 'cbow', 'window': 2, 'vector_size': 300},
    {'model_type': 'skipgram', 'window': 2, 'vector_size': 300},
    {'model_type': 'cbow', 'window': 4, 'vector_size': 300},
    {'model_type': 'skipgram', 'window': 4, 'vector_size': 300}
]

# Model eğitimi
for param in parameters:
    sg = 1 if param['model_type'] == 'skipgram' else 0

    # Lemmatized model
    model_lem = Word2Vec(
        sentences=df['metin_lemmatized'],
        vector_size=param['vector_size'],
        window=param['window'],
        sg=sg,
        min_count=1,
        workers=4
    )
    model_lem.save(f"word2vec_lemmatized_{param['model_type']}_win{param['window']}_dim{param['vector_size']}.model")

    # Stemmed model
    model_stem = Word2Vec(
        sentences=df['metin_stemmed'],
        vector_size=param['vector_size'],
        window=param['window'],
        sg=sg,
        min_count=1,
        workers=4
    )
    model_stem.save(f"word2vec_stemmed_{param['model_type']}_win{param['window']}_dim{param['vector_size']}.model")
