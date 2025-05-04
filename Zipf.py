import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np
import re

# 1. CSV dosyasını yükle
df = pd.read_csv(r"C:\Users\selma\Desktop\e-ticaret_urun_yorumlari.csv", sep=';')

# 2. Yorumları tek metin haline getir
texts = df['Metin'].astype(str).tolist()
full_text = " ".join(texts)

# 3. Temizleme işlemi: küçük harfe çevir, noktalama ve rakamları kaldır
cleaned = re.sub(r"[^a-zA-ZçÇğĞıİöÖşŞüÜ\s]", "", full_text.lower())

# 4. Kelimelere ayır
words = cleaned.split()

# 5. Kelime sıklıklarını hesapla
word_counts = Counter(words)
frequencies = [freq for _, freq in word_counts.most_common()]
ranks = np.arange(1, len(frequencies) + 1)

# 6. Zipf log-log grafiğini çiz
plt.figure(figsize=(10, 6))
plt.plot(np.log(ranks), np.log(frequencies), color='orange')
plt.xlabel("log(Sıra)")
plt.ylabel("log(Frekans)")
plt.title("Zipf Yasası - Ham Veri (Ürün Yorumları)")
plt.grid(True)
plt.show()
