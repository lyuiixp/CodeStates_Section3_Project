from pymongo import MongoClient
import certifi
import pprint
import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


HOST = 'cluster0.xn7k0.mongodb.net'
USER = 'yoyong'
PASSWORD = 'qwe123'
DATABASE_NAME = 'myFirstDatabase'
COLLECTION_NAME = 'baseball_hitter_records_3'
MONGO_URI = f"mongodb+srv://{USER}:{PASSWORD}@{HOST}/{DATABASE_NAME}?retryWrites=true&w=majority"

client = MongoClient({MONGO_URI}, tlsCAFile=certifi.where())
database = client[DATABASE_NAME]
collection = database[COLLECTION_NAME]

data=[]
for x in collection.find():
    data.append(x)

df_fetch = pd.DataFrame.from_dict(data).drop(columns =['_id'],axis=1)
df_fetch = df_fetch[df_fetch.연봉_만원 <=150000]  
df_fetch= df_fetch[df_fetch != "  "]   # 빈칸 데이터 삭제
df_fetch.dropna(axis=0, inplace =True)

numeric_columns = df_fetch.columns[2:]
df_x = df_fetch[numeric_columns].apply(pd.to_numeric)     
df_y = df_fetch[['이름']]
df_final = pd.concat([df_x,df_y],axis=1)
df_final.reset_index(drop=True, inplace=True)

df_final1 = df_final[df_final.columns.difference(['이름'])]

target = '연봉_만원'
# 데이터 분리
X_vif = df_final1[['타수', '타율', '홈런', '병살', '삼진']]
Y_vif = df_final1[target]
X_train_vif, X_test_vif, y_train_vif, y_test_vif = train_test_split(X_vif, Y_vif, test_size=0.2, random_state=2)

model_pkl1 = LinearRegression()
model_pkl1.fit(X_train_vif, y_train_vif)

joblib.dump(model_pkl1, 'C:/Users/Yong/코드스테이츠 AI/Section Project/section3/flask_app/model/model.pkl')