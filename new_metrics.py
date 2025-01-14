from warnings import filterwarnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, GridSearchCV, cross_validate
from sklearn.preprocessing import LabelEncoder
from wordcloud import WordCloud



filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None )
pd.set_option('display.float_format', lambda x: '%.2f' % x)


df = pd.read_excel(r'C:\Users\PC\PycharmProjects\GooglePlay\Play\son.xlsx')

df2= pd.DataFrame({"Name":(pd.value_counts(df['Name']).index)})

df2.to_excel('Names.xlsx') #Doldurmak için excel dosyalarını dışa aktarırım

df2= pd.read_excel(r'C:\Users\PC\PycharmProjects\GooglePlay\Play2\pythonProject9\Names.xlsx') #İşlediğ-im dosyayı içe aktarırım

df['Cinsiyet'] = df['Name'].apply(lambda x: df2.loc[df2['Name'] == x, 'Cinsiyet'].values[0] if x in df2['Name'].values else None) # İsimlerden elde ettiğim cinsiyet verilerini database ime aktarırım.

df.groupby(['Cinsiyet','BERT_label','Recency_degree']).agg({'BERT_label':'count'})
#Erkekler daha memnunsuz ve en negatif yorumlar Regular zamanda oluyor.

df['title'].nunique() #3586 kişi yorum yapmış
df['title'].shape[0] #3672 yorum yapıldı.

#86 kişi tekrar yorum yapmış.

df3= pd.DataFrame({"Name":pd.value_counts(df['title']).index,"ID":np.arange(0,3586)}) #Nicknamelere göre IDler atarım.



for i in df3['Name']:
    if df['Name'].any()==i:
        print(df3.loc[df3['Name']==i,'ID'].values[0])


df['ID'] = df['title'].apply(lambda x: df3.loc[df3['Name'] == x, 'ID'].values[0] if x in df3['Name'].values else None) #Bu ID leri de title aracıyla atarım.

df['response'].value_counts()

df.to_excel('Temiz.xlsx')
df=pd.read_excel('Temiz.xlsx')

#Baştan title kadar gidecek ,iso_date gidecek,Name,yorum gidecek.Polsrity_score dan polarity_label2 ye kadar gidecek,negative_count,total_count
#new_polarity_score
df.columns

df.drop(columns=['Unnamed: 0.1', 'Unnamed: 0', 'id', 'title','iso_date','Name_count', 'yorum','polarity_score',
       'polarity_label', 'polarity_score2', 'polarity_label2','negative count', 'positive_count', 'total_count',
       'new_polarity_score','Name'],inplace=True) #Geçersiz kolonları attım ancak bu işlemi daha düzenli yapmakta fayda var



df['response_by_comp'] = df['response'].apply(lambda x: 1 if isinstance(x, str) else 0) #Şirketin geri bildirim verdiği yorumları 1, geri bildirim vermediği yorumları 0 olarak belirtirim.


df['like_std']= pd.cut(df['likes'],5,labels=[1,2,3,4,5]) #Yorumların beğeni sayılarını 5 e göre standartlaştırırım.






df['Recency_degree'].value_counts()
df.loc[df['Recency_degree']=='New','rec_std']=5
df.loc[df['Recency_degree']=='Normal','rec_std']=4
df.loc[df['Recency_degree']=='Regular','rec_std']=3
df.loc[df['Recency_degree']=='Old','rec_std']=2

df['like_std'] = df['like_std'].astype(int)

df['response_by_comp'] = df['response_by_comp'].astype(int)
df['rec_std'] = df['rec_std'].astype(int)




#Burası Özel bir kısım
df['score_wtho_time'] = df['like_std']*0.9+df['response_by_comp']*0.1

df[df['score_wtho_time']>1.9] #Bu sisteme göre değerli yorumların ne zaman başladığını belirlerim.

df['score']=df['rec_std']*0.4+df['like_std']*0.5+df['response_by_comp']*0.1

df.loc[df['score']>3,:] #Buna bağlı olarak hangi değer aralığında olanaların iyi yorumda olduklarını karşılaştırırım.

#BERT e Label Encoder işlemi yaparım
from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()

df['le_BERT']= le.fit_transform(df['BERT_label']).astype(int) #BERT labellarını pozitifse 1 negatifse 0 olacak şekilde label encoder yaparım.



df['begeni_puanı']=df['le_BERT']*5+df['rating']*0.2 #Buna bağlı olarak bir değişken tasarlarım.

df['begeni_puanı'].describe()


df.to_excel('son.xlsx')


