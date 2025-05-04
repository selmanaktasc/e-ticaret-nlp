import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np
import ast

# Dosyayı yükle
df = pd.read_csv("cleaned_with_lemmatization.csv")

# Dize olarak gelen listeyi tekrar listeye çevir (çünkü CSV'de string olarak kaydedilmiş olabilir)
df['metin_lemmatized'] = df['metin_lemmatized'].apply(ast.literal_eval)

# Tüm kelimeleri birleştir
lemmatized_words = [word for tokens in df['metin_lemmatized'] for word in tokens]

# Frekans hesapla
freq = Counter(lemmatized_words)
sorted_freq = sorted(freq.values(), reverse=True)

# Zipf grafiği
plt.figure(figsize=(10,6))
plt.plot(np.log(range(1, len(sorted_freq)+1)), np.log(sorted_freq), color='green')
plt.xlabel("log(Sıra)")
plt.ylabel("log(Frekans)")
plt.title("Zipf Grafiği - Lemmatization Sonrası")
plt.grid()
plt.show()
