#!/usr/bin/env python
# coding: utf-8

# # Prodigy Infotech

# # Author : Ankita Dwivedi
# 

# Data Science

# Task-04

# Task : Analyze and visualize sentiment patterns in social media data to understand public opinion and attitudes towards specific topics or brands.

# DataSet Link:  https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import os
     


# In[9]:


col = ['Id', 'Entity', 'Sentiment', 'Content']
file_path = r"C:\Users\Ankita Dwivedi\Downloads\archive (2)\twitter_training.csv"

df_train = pd.read_csv(file_path, names=col)


# In[11]:


col = ['Id', 'Entity', 'Sentiment', 'Content']
file_path = r"C:\Users\Ankita Dwivedi\Downloads\archive (2)\twitter_validation.csv"

df_test = pd.read_csv(file_path, names=col)


# In[12]:


df_train


# # Data Summary

# In[13]:


df_train.shape


# In[14]:


df_train.columns


# In[15]:


df_train.info()


# In[16]:


df_train.dtypes


# # Data Cleaning

# In[17]:


df_train.isnull().sum()


# In[18]:


df_train.dropna(subset=['Content'] , inplace=True)


# In[19]:


df_train.shape


# In[20]:


df_train.Sentiment.unique()


# In[21]:


df_train.Sentiment=df_train.Sentiment.replace('Irrelevant' , 'Neutral')
df_test.Sentiment=df_test.Sentiment.replace('Irrelevant' , 'Neutral')


# In[23]:


df_train.Sentiment.unique()


# # EDA - Explorartory Data Analysi

# In[24]:


sentiment_count=df_train.Sentiment.value_counts()
sentiment_count


# In[25]:


y=['Neutral' , 'Negative' , 'Positive']
plt.pie(sentiment_count , labels=y, autopct='%0.1f%%' )
circle=plt.Circle((0,0),0.4, facecolor='white')
plt.gca().add_patch(circle)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()


# In[26]:


df_train.Entity.unique()


# In[27]:


Entity_count=df_train.Entity.value_counts()
Entity_count


# In[28]:


Entity_sort=Entity_count.sort_values(ascending=False)


# In[29]:


Entity_top10=Entity_sort.head(10)
Entity_top10


# In[30]:


Entity_index=Entity_top10.index


# In[33]:


plt.figure(figsize=(13,5))

x=['ApexLegends' , 'WorldOfCraft' , 'Dota2' , 'Microsoft' , 'Facebook' , 'TomClancysRainbowSix' , 'Verizon' , 'CallOfDuty' , 'LeagueOfLegends' , 'MaddenNFL']
y=[2353,2357,2359,2361,2362,2364,2365,2367,2377,2377]

plt.bar( x , y , alpha=0.7 , color='red')

for i,v in enumerate(y):
    plt.text(i,v,str(v),ha='center',weight='bold' )

plt.xticks(rotation=45)
plt.xlabel('Entity')
plt.ylabel('Number of Post in twitter')
plt.show()


# In[34]:


Entity_top3_df=Entity_sort.head(3)
Entity_top3_df


# In[35]:


Entity_top3=Entity_top3_df.index.tolist()
Entity_top3


# In[36]:


sentiment_by_entity=df_train.loc[df_train['Entity'].isin(Entity_top3)].groupby('Entity')['Sentiment'].value_counts().sort_index()
sentiment_by_entity


# # Model

# In[38]:


plt.figure(figsize=(10, 5))

y = ['Neutral', 'Negative', 'Positive']
color = ['#9C6383', '#839C63', '#63839C']

# Pie chart 1
plt.subplot(1, 3, 1)
plt.pie(sentiment_by_entity[:3], labels=y, autopct='%0.1f%%', textprops={'fontsize': 10}, colors=color)
plt.legend(labels=y, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10, title='Sentiment', title_fontsize=10)

# Pie chart 2
plt.subplot(1, 3, 2)
plt.pie(sentiment_by_entity[3:6], labels=y, autopct='%0.1f%%', textprops={'fontsize': 10}, colors=color)
plt.legend(labels=y, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10, title='Sentiment', title_fontsize=10)

# Pie chart 3
plt.subplot(1, 3, 3)
plt.pie(sentiment_by_entity[6:], labels=y, autopct='%0.1f%%', textprops={'fontsize': 10}, colors=color)
plt.legend(labels=y, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10, title='Sentiment', title_fontsize=10)

plt.tight_layout()
plt.show()


# In[39]:


df_train


# In[40]:


df_train.drop(['Id'] , axis=1 , inplace=True)


# In[41]:


df_test.drop(['Id'] , axis=1 , inplace=True)


# In[42]:


#train test split
X_train=df_train.drop(['Sentiment'] , axis=1)
X_test=df_test.drop(['Sentiment'] , axis=1)
y_train=df_train['Sentiment']
y_test=df_test['Sentiment']


# In[43]:


df_train.Sentiment.unique()


# In[44]:


#count the no of words in a sentence
from sklearn.feature_extraction.text import CountVectorizer


# In[45]:


v=CountVectorizer()
X_train_count=v.fit_transform(X_train.Content)
   


# In[46]:


#label Encoding
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
y_train=le.fit_transform(y_train)
y_test=le.fit_transform(y_test)


# In[47]:


y_train


# In[48]:


X_train.drop(['Entity'],axis=1,inplace=True)
X_test.drop(['Entity'],axis=1,inplace=True)


# In[49]:


#model
from sklearn.naive_bayes import MultinomialNB
model=MultinomialNB()
model.fit(X_train_count,y_train)


# In[50]:


comment=[
    'I am coming to the borders and I will kill you.'
]
comment_count=v.transform(comment)
model.predict(comment_count)


# In[51]:


X_test_count=v.transform(X_test.Content)
X_test_count.toarray()


# In[52]:


X_test_count.shape


# In[53]:


#score
model.score(X_test_count,y_test)


# In[ ]:




