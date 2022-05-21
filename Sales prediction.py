'''import my libraries '''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics
import warnings 
warnings.filterwarnings('ignore')

item_categories = pd.read_csv('E://Data Science//Training//Datasets//Future sales//competitive-data-science-predict-future-sales//item_categories.csv')
items = pd.read_csv('E://Data Science//Training//Datasets//Future sales//competitive-data-science-predict-future-sales//items.csv')
traindf = pd.read_csv('E://Data Science//Training//Datasets//Future sales//competitive-data-science-predict-future-sales//sales_train.csv')
shops = pd.read_csv('E://Data Science//Training//Datasets//Future sales//competitive-data-science-predict-future-sales//shops.csv')
test = pd.read_csv('E://Data Science//Training//Datasets//Future sales//competitive-data-science-predict-future-sales//test.csv')
sample_submission = pd.read_csv('E://Data Science//Training//Datasets//Future sales//competitive-data-science-predict-future-sales//sample_submission.csv')

'''time'''
traindf['date'] = pd.to_datetime(traindf['date'], errors='coerce')
traindf['Year'] = traindf['date'].dt.year
traindf['Month'] = traindf['date'].dt.month
traindf['day'] = traindf['date'].dt.day

traindf.loc['2013-2': , 'shop_id'].plot()
traindf.loc['2013-2':'2014-2', 'item_id'].plot()

# getting rid of "!" before shop_names
shops['shop_name'] = shops['shop_name'].map(lambda x: x.split('!')[1] if x.startswith('!') else x)
shops['shop_name'] = shops["shop_name"].map(lambda x: 'СергиевПосад ТЦ "7Я"' if x == 'Сергиев Посад ТЦ "7Я"' else x)
shops['city'] = shops['shop_name'].map(lambda x: x.split(" ")[0])
# lets assign code to these city names too
shops['city_code'] = shops['city'].factorize()[0]
traindf = pd.merge(traindf, shops, on="shop_id", how="inner")

traindf = pd.merge(traindf, items, on="item_id", how="inner")

cat_list = []
for name in item_categories['item_category_name']:
    cat_list.append(name.split('-'))
item_categories['split'] = (cat_list)
item_categories['cat_type'] = item_categories['split'].map(lambda x: x[0])
item_categories['cat_type_code'] = item_categories['cat_type'].factorize()[0]
item_categories['sub_cat_type'] = item_categories['split'].map(lambda x: x[1] if len(x)>1 else x[0])
item_categories['sub_cat_type_code'] = item_categories['sub_cat_type'].factorize()[0]
item_categories.drop('split', axis = 1, inplace=True)
traindf = pd.merge(traindf, item_categories, on="item_category_id", how="inner")

traindf['revenue'] = traindf['item_price'] *  traindf['item_cnt_day']

traindf.drop(['item_name', 'shop_name', 'city', 'item_category_name', 'cat_type', 'sub_cat_type'], axis = 1, inplace=True)

fig,ax = plt.subplots(2,1,figsize=(10,4))

plt.xlim(-300, 3000)
ax[0].boxplot((traindf.item_cnt_day) , labels=['train.item_cnt_day'], vert=False)
plt.xlim(-1000, 350000)
ax[1].boxplot((traindf.item_price) , labels=['train.item_price'], vert=False)
plt.show()

traindf = traindf[traindf.item_price<100000]
traindf = traindf[traindf.item_cnt_day<1001]


traindf = traindf.groupby(['date_block_num', 'shop_id', 'item_id' ])["date","item_price","item_cnt_day"].agg({"date":["min",'max'],"item_price":"mean","item_cnt_day":"sum"})
traindf.rename(columns={'item_cnt_day': 'item_cnt_month'}, inplace=True)


traindf = traindf[traindf['shop_id'].isin(test['shop_id'].unique())]
traindf = traindf[traindf['item_id'].isin(test['item_id'].unique())]


# final_train_df = pd.merge(test,final_train_df,on = ['item_id','shop_id'],how = 'left')
# final_train_df.fillna(0,inplace = True)

# grouped = pd.DataFrame(traindf.groupby(['year','month']))
# sns.pointplot(x='month', y='item_cnt_day', hue='year', data=grouped)
