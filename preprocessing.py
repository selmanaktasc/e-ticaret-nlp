import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
from trnlp import TrnlpWord
from nltk.stem.porter import PorterStemmer

# Gerekli indirmeler
nltk.download('stopwords')

# Veriyi oku ve ilk 5000 satırı al
df = pd.read_csv(r"C:\Users\selma\Desktop\e-ticaret_urun_yorumlari.csv", sep=';')
df = df.head(5000)

# --- 1. Lowercasing ---
df['metin_lower'] = df['Metin'].astype(str).str.lower()

# --- 2. HTML ve özel karakter temizliği ---
def clean_html(text):
    return BeautifulSoup(text, "html.parser").get_text()

df['metin_cleaned'] = df['metin_lower'].apply(clean_html)

# --- 3. Noktalama temizliği + Tokenization ---
def tokenize(text):
    text = re.sub(r"[^a-zA-ZçÇğĞıİöÖşŞüÜ\s]", "", text)
    return text.split()

df['metin_tokenized'] = df['metin_cleaned'].apply(tokenize)

# --- 4. Stopword Removal ---
stop_words = set(stopwords.words('turkish'))

def remove_stopwords(tokens):
    return [word for word in tokens if word not in stop_words]

df['metin_nostop'] = df['metin_tokenized'].apply(remove_stopwords)

# --- 5. Lemmatization ---
def lemmatize(tokens):
    stems = []
    for token in tokens:
        analyzer = TrnlpWord()
        analyzer.setword(token)
        stems.append(analyzer.get_stem)  
    return stems


df['metin_lemmatized'] = df['metin_nostop'].apply(lemmatize)


# --- 6. Stemming ---
stemmer = PorterStemmer()

def stem_tokens(tokens):
    return [stemmer.stem(token) for token in tokens]

df['metin_stemmed'] = df['metin_lemmatized'].apply(stem_tokens)

# --- Örnek çıktıları göster (ilk 5 satır)
for i in range(10):
    print(f"\n{i+1}. CÜMLE:")
    print(f"Orijinal        : {df['Metin'][i]}")
    print(f"Lowercased      : {df['metin_lower'][i]}")
    print(f"Temizlenmiş     : {df['metin_cleaned'][i]}")
    print(f"Tokenized       : {df['metin_tokenized'][i]}")
    print(f"Stopwords Çıkar : {df['metin_nostop'][i]}")
    print(f"Lemmatized      : {df['metin_lemmatized'][i]}")
    print(f"Stemmed         : {df['metin_stemmed'][i]}")
    print("-" * 100)

df[['Metin', 'metin_lemmatized']].to_csv("cleaned_with_lemmatization.csv", index=False)
df[['Metin', 'metin_stemmed']].to_csv("cleaned_with_stemming.csv", index=False)
df.to_csv("e-ticaret_urun_yorumlari_cleaned.csv", index=False)
