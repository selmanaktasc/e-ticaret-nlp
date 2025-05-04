import pandas as pd
import ast

# Lemmatization dosyası
df_lemma = pd.read_csv("cleaned_with_lemmatization.csv")
df_lemma['metin_lemmatized'] = df_lemma['metin_lemmatized'].apply(ast.literal_eval)
lemmatized_words = [word for tokens in df_lemma['metin_lemmatized'] for word in tokens]

# Stemming dosyası
df_stem = pd.read_csv("cleaned_with_stemming.csv")
df_stem['metin_stemmed'] = df_stem['metin_stemmed'].apply(ast.literal_eval)
stemmed_words = [word for tokens in df_stem['metin_stemmed'] for word in tokens]

# İstatistikler
print("📊 Lemmatization Verisi")
print("Toplam kelime sayısı:", len(lemmatized_words))
print("Benzersiz kelime sayısı:", len(set(lemmatized_words)))

print("\n📊 Stemming Verisi")
print("Toplam kelime sayısı:", len(stemmed_words))
print("Benzersiz kelime sayısı:", len(set(stemmed_words)))
