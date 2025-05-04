import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import ast

# Dosyayı oku (önceki preprocessing sonunda oluşturulmuş CSV)
df = pd.read_csv("e-ticaret_urun_yorumlari_cleaned.csv")

# Listeleri düz metne çevir
df['lemmatized_text'] = df['metin_lemmatized'].apply(ast.literal_eval).apply(lambda x: ' '.join(x))
df['stemmed_text'] = df['metin_stemmed'].apply(ast.literal_eval).apply(lambda x: ' '.join(x))

# TF-IDF: Lemmatized
vectorizer_lem = TfidfVectorizer()
tfidf_lem = vectorizer_lem.fit_transform(df['lemmatized_text'])
df_tfidf_lem = pd.DataFrame(tfidf_lem.toarray(), columns=vectorizer_lem.get_feature_names_out())
df_tfidf_lem.to_csv("tfidf_lemmatized.csv", index=False)

# TF-IDF: Stemmed
vectorizer_stem = TfidfVectorizer()
tfidf_stem = vectorizer_stem.fit_transform(df['stemmed_text'])
df_tfidf_stem = pd.DataFrame(tfidf_stem.toarray(), columns=vectorizer_stem.get_feature_names_out())
df_tfidf_stem.to_csv("tfidf_stemmed.csv", index=False)

print("✅ TF-IDF dosyaları başarıyla oluşturuldu ve kaydedildi.")
