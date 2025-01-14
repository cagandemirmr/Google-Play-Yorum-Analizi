
from warnings import filterwarnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer


filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None )
pd.set_option('display.float_format', lambda x: '%.2f' % x)

##############################################################
#EDA
##############################################################


df = pd.read_excel(r'C:\Users\PC\PycharmProjects\GooglePlay\google_play_reviews.xlsx') #Dosyayı okuturum

df.drop(columns='avatar',inplace=True) #Gereksiz dosyayı silerim.

df.shape #Toplam 3672 satır ve 8 sütun var.
df.isna().sum() #Sadece Response değerinde 3666 satırlık veri eksik

df.loc[:,'Name']= df['title'].apply(lambda x : [word.lower() for word in x.split(' ')][0]) #Adlarını ayırırım ve küçük harfle yazdırırım.
df['Name'] = df['Name'].str.replace('[^\w\s]','',regex=True) #Özel karakterlerden ayırırım.

df.loc[:,'Name_count']= df['Name'].apply(lambda x : len(x)) # İsimler için karakter sayısı için bir kolon oluştururum

df['snippet'] = df['snippet'].apply(lambda x:str(x).lower()) #Yorumların harflerini küçültürüm.
df['snippet'] = df['snippet'].str.replace('[^\w\s]','',regex=True) #Yorumlardan noktalama işaretlerini artar.
df['snippet'] = df['snippet'].str.replace('\d', '',regex=True) #Yorumlardan sayısal değerleri çıkartırız.

df['yorum']=df['snippet'].copy() #Bu kolonu daha sonra kullanacağım o yüzden snippet kolonu üzerinden kopya aldım.



import re  #Emojileri silmek için bir fonksiton yazarım

def remove_emojis(text):
    # Emoji desenini tanımlama
    emoji_pattern = re.compile(
        "["        
        "\U0001F600-\U0001F64F"  # Smileys
        "\U0001F300-\U0001F5FF"  # Symbols & Pictographs
        "\U0001F680-\U0001F6FF"  # Transport & Map Symbols
        "\U0001F700-\U0001F77F"  # Alchemical Symbols
        "\U0001F780-\U0001F7FF"  # Geometric Shapes
        "\U0001F800-\U0001F8FF"  # Supplemental Arrows
        "\U0001F900-\U0001F9FF"  # Supplemental Symbols
        "\U0001FA00-\U0001FA6F"  # Chess Symbols
        "\U00002700-\U000027BF"  # Miscellaneous Symbols
        "\U00002600-\U000026FF"  # Miscellaneous Symbols
        "\U000024C2-\U0001F251"  # Enclosed Characters
        "]+",
        flags=re.UNICODE
    )
    return emoji_pattern.sub(r'', text)

df['Name'] = df['Name'].apply(lambda x: remove_emojis(x)) #Emojileri silerim.

df.loc[df['Name_count']<3,'Name'].unique() #2 den az olan isimlerin X ve B olduğunu gördüm

len(list(df.loc[df['Name_count'] > 2,'Name'].unique())) # 1527 unique isim varmış

isimler = list(df.loc[df['Name_count'] > 2,'Name'].unique()) #Bu isimleri listeme alırım. İsmi su olanlardan özür diliyorum ancak istisnalar kaideyi bozmaz.

df['date']= pd.to_datetime(df['date']) #Veri Tipini DATETIME a çeviririm.

df['Recency'] = (pd.to_datetime('2024-12-18')- df['date']).dt.days #Verileri çektiğim günden belirli bir süre sonrasına tarih belirleyip Recency değerini belirlerim.


#Recency lerini date e göre değiştiririm.
df[df['Recency']  <= 7].shape[0] #New 1661
df[(df['Recency']  > 7) & (df['Recency']  <= 30)].shape[0] #Normal  9815
df[(df['Recency']  > 30) & (df['Recency']  <= 90)].shape[0] #Regular  17063
df[df['Recency']> 90].shape[0] #Old 1510



df.loc[df['Recency']  <= 7,'Recency_degree']= 'New'
df.loc[(df['Recency']  > 7) & (df['Recency']  <= 30),'Recency_degree']= 'Normal'
df.loc[(df['Recency']  > 30) & (df['Recency']  <= 90),'Recency_degree']= 'Regular'
df.loc[df['Recency']> 90,'Recency_degree'] = 'Old'


Recency_by_rating= df.groupby(['Recency_degree']).agg({'rating':'mean'}) #Recency' e göre işlem yapabiliriz.
'''                rating
Recency_degree        
New               3.96
Normal            3.97
Old               4.12
Regular           4.01
'''

#############################################################
#Metin İşleme
#############################################################

import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

# Türkçe stopword listesini alırım.
turkish_stopwords = set(stopwords.words('turkish'))

df['snippet'] = df['snippet'].apply(lambda x: " ".join(x for x in str(x).split() if x not in turkish_stopwords)) #Stopwordslerden kurtuluruz.

temp_df =pd.Series(''.join(df['snippet']).split()).value_counts() #Çok düşük frekansta olan değerleri saptarım.

drops=list(temp_df[temp_df<=1].index) 

df['snippet']= df['snippet'].apply(lambda x: ' '.join(x for x in str(x).split() if x not in drops)) #Bir kelime olan kelimelerden kurtulurum.

'''import spacy

nlp = spacy.load("tr_core_news_sm")
doc = nlp("Merhaba dünya!")
for token in doc:
    print(token.text, token.lemma_) '''

from zeyrek import MorphAnalyzer #Terminale Conda install pip dyazmak gerekiyor.

import nltk

nltk.download('punkt_tab')

analyzer = MorphAnalyzer()


df['lemma']= df['snippet'].apply(lambda x: ' '.join(analyzer.analyze(x)[0][0][1] for x in str(x).split())) #Morph analyzerla kök almaya çalıştım ama sonuç hayal kırıklığı oldu.Türkçede malesef işe yaramıyor





#######################
nltk.download("vader_lexicon")
sia = SentimentIntensityAnalyzer() #NLTK ya göre Sentiment analizi yaptım ama Türkçede malesef başarılı değil.

df['polarity_score'] = df['lemma'].apply(lambda x: sia.polarity_scores(x)['compound'])

df.loc[df['polarity_score']==0,'polarity_label'] = 'nötr'
df.loc[df['polarity_score']> 0,'polarity_label'] = 'positive'
df.loc[df['polarity_score'] < 0,'polarity_label'] = 'negative'

df['polarity_label'].value_counts()

'''nötr        3582
positive      80
negative      10'''
######################################################
df['polarity_score2'] = df['lemma'].apply(lambda x: sia.polarity_scores(x)['compound'])

df.loc[df['polarity_score2']==0,'polarity_label2'] = 'nötr'
df.loc[df['polarity_score2']> 0,'polarity_label2'] = 'positive'
df.loc[df['polarity_score2'] < 0,'polarity_label2'] = 'negative'

df['polarity_label2'].value_counts() #Lemma yapmak polarity i çok değiştiremedi çünkü NLTK türkçe uygun değil.
'''polarity_label2
nötr        3582
positive      80
negative      10'''



df[df['rating']==3].shape[0] #222 Nötr
df[df['rating'] > 3].shape[0] #2783 Positive
df[df['rating'] < 3].shape[0] #667 Negative

# Aslında çoğu kişi beğenmiş ama düşük not vermiş ve birbiriyle çelişen durumlar da söz konusu bunun nedeni NLTK nın Türkçeye uyumlu olmaması

df.loc[df['rating']==3,'status'] = 'nötr'
df.loc[df['rating'] > 3, 'status'] = 'positive'
df.loc[df['rating'] < 3, 'status'] = 'negative'

rating_sentiment = df.groupby('status').count()['polarity_label']

'''
status
negative     667
nötr         222
positive    2783
'''
##########################################################
# Sonuçları test etme
###########################################################



df['title'].unique().size # toplam 3586 kişi yorum yapmış.

#86 adet tekrarlayan yorum var.

#Rating ile Polarity Score un örtüştüğü Yorumlara bakarım

negative = list(df[(df['status']=='negative' )& (df['polarity_label']=='negative')].index)


positive = list(df[((df['status']=='positive' )& (df['polarity_label']=='positive'))].index)

notr = list(df[((df['status']=='nötr' )& (df['polarity_label']=='nötr'))].index)

normal = negative + positive + notr #Standart olanları elerim.

df.loc[normal,'snippet'].unique().size #Toplamda 272 tane polarity score la eşleşen rating değeri var.

list(df.loc[negative,'snippet'])

list(df.loc[positive,'snippet'])

list(df.loc[notr,'snippet'])



df.loc[notr,'snippet'].to_excel('nötr.xlsx')


adf = df.loc[~df.index.isin(normal), :] #Kinayeli olanları bir araya getiririm.

adf.loc[adf['polarity_label']=='negative','snippet'].unique().size #7 adet yoruma rastlarım

adf.loc[adf['polarity_label']=='positive','snippet'].unique().size #9 adet yoruma rastlarım

adf.loc[adf['polarity_label']=='nötr','snippet'].unique().size # 2604 adet yoruma rastlarım


277+2715+13+12 # Buna bağlı olarak aslında bizim 3017 adet benzersiz yorumumuz var.

df['title'].nunique() #3586 adet unique kullanıcımız var.

list(adf.loc[(adf['polarity_label']=='positive') & (adf['status']=='negative'),'yorum'])[0]
''''oyun çok güzel fakat joystick sorunu var önceden yoktu
 böyle birşey ileri veya geri giderken yönlendirmek çok zor oluyor güncelleme ile lütfen bu sorunu çözün'''


list(adf.loc[(adf['polarity_label']=='negative') & (adf['status']=='positive'),'yorum'])[2]
'''çok iyi bir oyun ama yerinizi geniştir mek çok zor tek sorun bu başka problem yok çok güzel bir oyum indirme nizi tamsiye ederim'''

tf = (df['snippet'].apply(lambda x: pd.value_counts(x.split(" "))).sum(axis = 0).reset_index())

tf.columns = ["words","Count"]

tf.sort_values("Count",ascending=False).head(100)


#####################################################
#Kendi Polarity Score umu oluştururum
#####################################################


negative_words = [                                       #Bunun için önceki bölümde hem polarity score u 3 altında hem de polarity score u negatif olan yorumlardan kötü olarak algılanan kelimeleri alırım.
    "zor", "sıkıcı", "sorun", "fazla", "kötü", "eksik",
    "yetersiz", "şikayet", "saçma", "problem", "monoton",
    "yavaş", "pahalı", "düşük", "gelişmemiş",
    "kısıtlı", "etkili değil", "beklemek", "süre",
    "teslimat", "can sıkıcı", "sıkıntı", "dengesiz",
    "kapsamlı değil", "kullanıcı dostu değil",
    "bozuk", "sorunlu", "tahmin edilebilir", "kuru",
    "tekrarlayan", "dar", "ilgisiz", "sonradan", "ancak",
    "ama", "fakat", "lakin", "sakın", "asla", "takılmak",
    "değil", "üzdü", "etme", "anlamsız", "karmaşık",
    "kafa karıştırıcı", "desteksiz", "hata", "kesinti",
    "kopma", "uzun", "işlevsiz", "yanlış", "tutarsız", "beklenti altında", "vasat",
    "tatmin edici değil", "çökme", "hantal", "gereksiz",
    "rahatsız edici", "dayanılmaz", "zaman alıcı",
    "geçersiz", "aceleye gelmiş",
    "yanıltıcı", "uyumsuz", "verimsiz",
    "uyarı", "zayıf", "güçsüz", "hüsran", "sabır gerektiren",
    "hayal kırıklığı", "düzensiz",
    "karmaşa", "bıktırıcı", "berbat", "olumsuz",
     "kopuk","enerji","biraz","aşırı","lazım,"
]


negative_words = [x.replace(x,analyzer.analyze(x)[0][0][1]) for x in negative_words] 




from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(vocabulary=negative_words) #Buna bağlı olarak vector analyzer ı negatif kelimelere göre ayarlama yaptırırım.

texts = df['lemma'].astype(str)  # Metinlerin string formatında olduğundan emin olurum.
X = vectorizer.fit_transform(texts)


print(X.toarray())

df['negative count']= [np.sum(x) for x in X] #Ve negatif olan kelime sayılarını bir cümlede toplarım.



#Aynı işlemi pozitif kelimeler için de yaparım.
positive_words = [
    "güzel", "mükemmel", "eğlenceli", "iyi", "keyifli",
    "hoş", "başarılı", "sevimli", "faydalı", "ilginç",
    "zevkli", "tavsiye", "gelişmiş", "harika",
    "yeni", "şık", "eğitimsel", "dinamik", "şaşırtıcı",
    "akıcı", "kullanıcı dostu", "renkli", "özgün",
    "sosyal", "yaratıcı", "etkileşimli",
    "ilham verici", "zengin içerik"
]

positive_words = [x.replace(x,analyzer.analyze(x)[0][0][1]) for x in positive_words]

vectorizer = CountVectorizer(vocabulary=positive_words)

pos = vectorizer.fit_transform(texts)


df['positive_count']= [np.sum(x) for x in pos]

df['total_count'] = (df['positive_count']) - (df['negative count']*1.5) #Buna bağlı olarak bir denklem oluştururum.


df.loc[(df['total_count'] < 0),'new_polarity_score'] = "negative"  #Çıkan sonuca göre 3 durumda belirtieim
df.loc[(df['total_count'] > 0),'new_polarity_score'] = "positive"
df.loc[(df['total_count'] == 0),'new_polarity_score'] = "nötr"



df.loc[df['new_polarity_score']=='negative','snippet'].shape[0] #998 negative
df.loc[df['new_polarity_score']=='positive','snippet'].shape[0] #1651 pozitif
df.loc[df['new_polarity_score']=='nötr','snippet'].shape[0] #1023


list(df.loc[df['new_polarity_score']=='negative','snippet'].sample(1))

######################################################################
#Yeni Oluşturulan Puanlama Sisteminde Negatif Yorumların Değerlendirilmesi
######################################################################

df.columns

df.loc[(df['new_polarity_score']=='negative')& (df['status']=='negative'),['snippet']].shape[0] #268 satır gerçekten kötü olanları yakaladım.

df.loc[(df['new_polarity_score']=='negative')& (df['status']=='positive'),'snippet'].shape[0] #446 oyun 3 ten büyük ancak yorumlarda gerçekten de olumsuzluk olan verileri yakaladım

df.loc[(df['new_polarity_score']=='negative')& (df['status']=='nötr'),'snippet'].shape[0] #98 tane negatif yorum nötr yorumlama durumuyla çatışıyor ancak yine de en mantıklısı yorumlara bakmak.


#Sonuç oyuncular her ne kadar sevdikleri oyunu öne çıkarmak için yüksek oy verse de çoğu zaman yorumu öne çıksın diye iyi oy verebilmektedir.



######################################################################
#Yeni Oluşturulan Puanlama Sisteminin Değerlendirilmesi
######################################################################


list(df.loc[(df['new_polarity_score']=='positive')& (df['status']=='negative'),'snippet'].sample(1))

#değil yerine deyil yazılması da ayrı bir sorun.

#Oluşturduğum "model" olumsuz kelimeleri saptayabiliyor ancak kinayeli yorumları algılayamıyor.


#######################################################################
#BERT Modelini İşin İçine Sokma
#######################################################################

from transformers import AutoModel, AutoTokenizer   #BERT modelini ve ilgili kütüphaneleri yüklerim

tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-turkish-cased")
model = AutoModel.from_pretrained("dbmdz/bert-base-turkish-cased")


df['BERT_label']= df['snippet'].apply(lambda x:sentiment_analyzer(x)[0]['label']) 

df.loc[(df['BERT_label']=='positive'),['snippet','BERT','BERT_label']]


df['comment_length']=df['snippet'].apply(lambda x: len(x))

df.loc[(df['BERT_label']=='negative') & (df['comment_length']<18) ,['polarity_label','polarity_label2','new_polarity_score','BERT_label','snippet','comment_length']].sample(1)

df['new_polarity_score'].value_counts()
'''positive    1671
nötr        1024
negative     977'''


df['BERT_label'].value_counts()
'''positive    2083
negative    1589'''


#müq müüüüük gibi kelimeleri algılamada çok zorluk yaşıyorlar.


df['comment_length'].describe()
'''
count   3672.00
mean      47.52
std       57.67
min        0.00
25%       12.00
50%       28.00
75%       58.00
max      391.00'''

#Ancak karşılaştırdığım zaman en iyi sonuç verenin BERT modeli olduğunu bu başarıyı izleyen diğer bir modelin ise benim kurduğum "model" olduğunu gördüm.
#Buna bağlı olarak belki de BERT Türkçe modelini buna göre eğitebilirim.



