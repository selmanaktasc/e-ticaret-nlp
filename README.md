# NLP Projesi: E-Ticaret Yorumları

Bu proje kapsamında, e-ticaret platformlarından alınan Türkçe ürün yorumları üzerinde doğal dil işleme (NLP) teknikleri uygulanmıştır. Amaç, yorumları ön işleme sürecinden geçirerek anlamlı bir biçimde vektörleştirip makine öğrenmesi için kullanılabilir hale getirmektir.

## 1. Veri Seti

* Kaynak: E-ticaret platformu yorumları
* Boyut: İlk 5000 yorum kullanılmıştır
* Format: CSV ("Metin" sütunu)

## 2. Ön İşleme (Pre-processing)

Aşağıdaki adımlar sırasıyla uygulanmıştır:

* Küçük harfe dönüştürme (Lowercasing)
* HTML ve özel karakter temizliği
* Tokenization (kelimeye ayırma)
* Stop word çıkarımı
* Lemmatization (kök indirgeme)
* Stemming (daha kaba kök indirgeme)

Her adımda çıkan veriler ayrı sütunlarda tutulmuş, örnek dönüşümler raporlanmıştır.

## 3. Temizlenmiş Veri Seti

* **cleaned\_with\_lemmatization.csv**
* **cleaned\_with\_stemming.csv**
* Zipf yasası doğrultusunda her iki veri seti için log-log grafikler çizilmiştir.

## 4. TF-IDF Vektörleştirme

* Her iki temizlenmiş veri seti için ayrı TF-IDF matrisleri oluşturulmuştur:

  * **tfidf\_lemmatized.csv**
  * **tfidf\_stemmed.csv**
* Satırlar belgeleri, sütunlar kelimeleri, hücreler TF-IDF değerlerini temsil etmektedir.

## 5. Word2Vec Vektörleştirme

* Gensim ile 16 farklı Word2Vec modeli eğitilmiştir.
* Kullanılan parametreler:

  * model\_type: CBOW ve Skip-gram
  * window: 2 ve 4
  * vector\_size: 100 ve 300
* Toplamda 8 lemmatized + 8 stemmed = 16 model dosyası oluşturulmuştur.
* Her modelin ismi açıkça parametreleri yansıtacak şekilde düzenlenmiştir.
* Örnek benzerlik sorguları raporda sunulmuştur.

## 6. Sonuç ve Değerlendirme

Lemmatization yöntemi, kelimelerin doğru köklerine ulaşarak anlamı daha iyi korumuştur. Stemming ise daha basit ve hızlı olsa da bazı anlam kayıplarına neden olmuştur. Word2Vec modellerinde Skip-gram daha başarılı sonuçlar verirken, CBOW hızlı ve etkili bir alternatif sunmuştur. Genel olarak, lemmatized + Skip-gram + 300 boyut kombinasyonu en başarılı sonuçları vermiştir.

---

Bu proje Türkçe dilinde NLP uygulamalarına giriş niteliğinde olup, elde edilen çıktılar başka görevlerde (ör. duygu analizi) kullanılmak üzere vektörel temsillere dönüştürülmüştür.
