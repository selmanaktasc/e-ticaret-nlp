import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np
import ast

# Stemming dosyasını oku
df = pd.read_csv("cleaned_with_stemming.csv")

# String olan listeleri tekrar listeye çevir
df['metin_stemmed'] = df['metin_stemmed'].apply(ast.literal_eval)

# Kelimeleri birleştir
stemmed_words = [word for tokens in df['metin_stemmed'] for word in tokens]

# Frekansları hesapla
freq = Counter(stemmed_words)
sorted_freq = sorted(freq.values(), reverse=True)

# Zipf grafiği
plt.figure(figsize=(10,6))
plt.plot(np.log(range(1, len(sorted_freq)+1)), np.log(sorted_freq), color='blue')
plt.xlabel("log(Sıra)")
plt.ylabel("log(Frekans)")
plt.title("Zipf Grafiği - Stemming Sonrası")
plt.grid()
plt.show()
