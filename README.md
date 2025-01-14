# Google-Play-Yorum-Analizi

Bu proje, Google Play Store uygulamalarına yönelik kullanıcı yorumlarını analiz ederek içgörüler elde etmeyi amaçlar. Çeşitli veri işleme ve analiz teknikleri kullanılarak, kullanıcı yorumlarının nitelikleri sınıflandırılmış, önemli metrikler hesaplanmış ve geri bildirimlerin anlamlı hale getirilmesi sağlanmıştır.

## Kullanılan Teknolojiler
- Python
- Pandas
- NumPy
- Matplotlib
- WordCloud
- NLTK
- Scikit-learn
- BERT

## Proje Adımları

### 1. Verilerin Google API ile çekilmesi
-Google'ın API ı kullanarak oyun yorumları,kullanıcı nicknameleri, oyuncu yorum tarihleri, yorum beğenileri, oyun oyları ve geri dönüşler gibi bir çok veri alındı.

### 2. Veri Hazırlığı

- Veri setindeki eksik değerler, gerekli eklemeler yapılarak temizlendi.
- Kullanıcıların cinsiyet bilgileri, isimlerine göre oluşturulan ek bir Excel dosyasından (`Names.xlsx`) çekilerek veri setine eklendi.
- Yorum yapan kullanıcıların benzersiz kimlikleri (ID'ler) oluşturularak, her kullanıcı için bir ID atanmıştır.

### 3. Verilerin Standartlaştırılması ve Yeni Değişkenler
- `likes` değişkeni 5'li kategorilere ayrılarak standartlaştırıldı (`like_std`).
- Yorumların zamanına göre derecelendirilmesi yapıldı (`rec_std`).
- Şirket geri bildirimi olan yorumlar için bir değişken oluşturuldu (`response_by_comp`).
- Kullanıcı yorumlarının genel değerini ölçmek için bir skor sistemi tasarlandı (`score` ve `score_wtho_time`).

### 4. BERT Modeli ve Sentiment Analizi
- Yorumların duygu analizi, BERT modelinden alınan etiketler (`BERT_label`) ile değerlendirildi.
- Bu etiketler, pozitif/negatif ayrımı için `LabelEncoder` kullanılarak sayısal verilere dönüştürüldü (`le_BERT`).
- BERT etiketleri ve puanlara dayalı olarak özel bir değişken oluşturuldu (`begeni_puanı`).

### 5. Çıktılar ve Öngörüler
- Veri setinde en yararlı yorumlar tespit edildi.
- Erkek kullanıcıların daha fazla olumsuz yorumlar yaparken kadın kullanıcıların daha az puan verdiği gözlenmiştir.
- Yorumların ve puanlar arasındaki ilişki incelendi ve zaman serisine göre incelendi.


## Notlar
- Kodda kullanılan bazı işlemler, veri setine özgüdür ve farklı veri kümelerine uygulanmadan önce uyarlanması gerekebilir.
- Şirket geri bildirimleri ve yorum zamanları gibi değişkenler, analiz sonuçlarının özelleştirilmesine yardımcı olmuştur.

## Gelecekteki Geliştirmeler
-BERT Türkçe modeli üzerinden daha spesifik bir model oluşturulacaktır.

