#!/usr/bin/env python
# coding: utf-8

# # Prodigy Infotech
# 

# # Author : Ankita Dwivedi
# Data Science

# Task-02

# Task: Perform data cleaning and exploratory data analysis (EDA) on a dataset of your choice, such as the Titanic dataset from Kaggle. Explore the relationships between variables and identify patterns and trends in the data.

# DataSet Link : https://www.kaggle.com/c/titanic/data

# In[7]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[8]:


plt.style.use('fivethirtyeight')
import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[9]:


train_data=pd.read_csv(r"C:\Users\Ankita Dwivedi\Downloads\train.csv")
test_data=pd.read_csv(r"C:\Users\Ankita Dwivedi\Downloads\test.csv")


# In[10]:


train_data.shape


# In[11]:


test_data.shape


# In[12]:


train_data.head(8)


# In[13]:


test_data.head(8)


# In[14]:


train_data.isnull().sum()


# In[21]:


sns.countplot(x='Survived', data=train_data)
plt.title('Survival Count')
plt.xlabel('Survived')
plt.ylabel('Count')
plt.show()


# In[16]:


train_data.groupby(['Sex', 'Survived'])['Survived'].count()


# In[20]:


sns.countplot(x='Sex', hue='Survived', data=train_data)
plt.title('Survival Count by Gender')
plt.xlabel('Sex')
plt.ylabel('Count')
plt.show()


# In[19]:


sns.countplot(x='Pclass', hue='Survived', data=train_data)
plt.title('Pclass: Survived vs Dead')
plt.xlabel('Pclass')
plt.ylabel('Count')
plt.show()


# In[15]:


pd.crosstab([train_data.Sex,train_data.Survived],train_data.Pclass,margins=True).style.background_gradient(cmap='summer_r')


# In[22]:


sns.catplot(x='Pclass', y='Survived', hue='Sex', data=train_data, kind='bar')
plt.title('Survival Rate by Pclass and Sex')
plt.xlabel('Pclass')
plt.ylabel('Survival Rate')
plt.show()


# In[23]:


print('Oldest person Survived was of:',train_data['Age'].max())
print('Youngest person Survived was of:',train_data['Age'].min())
print('Average person Survived was of:',train_data['Age'].mean())


# In[24]:


f, ax = plt.subplots(1, 2, figsize=(18, 8))

# Bar plot for Pclass and Age vs Survived
sns.barplot(x='Pclass', y='Age', hue='Survived', data=train_data, ax=ax[0])
ax[0].set_title('PClass and Age vs Survived')
ax[0].set_yticks(range(0, 110, 10))

# Bar plot for Sex and Age vs Survived
sns.barplot(x='Sex', y='Age', hue='Survived', data=train_data, ax=ax[1])
ax[1].set_title('Sex and Age vs Survived')
ax[1].set_yticks(range(0, 110, 10))

plt.show()


# In[25]:


train_data['Initial']=0
for i in train_data:
    train_data['Initial']=train_data.Name.str.extract('([A-Za-z]+)\.') #extracting Name initials


# In[26]:


pd.crosstab(train_data.Initial,train_data.Sex).T.style.background_gradient(cmap='summer_r')


# In[27]:


train_data['Initial'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','Countess',
                               'Jonkheer','Col','Rev','Capt','Sir','Don'],['Miss',
                                'Miss','Miss','Mr','Mr','Mrs','Mrs','Other','Other','Other','Mr','Mr','Mr'],inplace=True)


# In[28]:


train_data.groupby('Initial')['Age'].mean()


# In[31]:


train_data.loc[(train_data.Age.isnull()) & (train_data.Initial=='Mr'),'Age']=33
train_data.loc[(train_data.Age.isnull()) & (train_data.Initial=='Mrs'),'Age']=36
train_data.loc[(train_data.Age.isnull()) & (train_data.Initial=='Master'),'Age']=5
train_data.loc[(train_data.Age.isnull()) & (train_data.Initial=='Miss'),'Age']=22
train_data.loc[(train_data.Age.isnull()) & (train_data.Initial=='Other'),'Age']=46


# In[32]:


train_data.Age.isnull().any()


# In[33]:


f,ax=plt.subplots(1,2,figsize=(20,20))
train_data[train_data['Survived']==0].Age.plot.hist(ax=ax[0],bins=20,edgecolor='black',color='red')
ax[0].set_title('Survived = 0')
x1=list(range(0,85,5))
ax[0].set_xticks(x1)
train_data[train_data['Survived']==1].Age.plot.hist(ax=ax[1],bins=20,edgecolor='black',color='green')
x2=list(range(0,85,5))
ax[1].set_xticks(x2)
ax[1].set_title('Survived = 1')
plt.show()


# In[35]:


sns.catplot(x='Pclass', y='Survived', hue='Initial', kind='bar', data=train_data, col='Initial', height=4, aspect=0.7)
plt.subplots_adjust(top=0.8)  # Adjust the top space to accommodate the title
plt.suptitle('Survived by Pclass and Initial')
plt.show()


# In[37]:


pd.crosstab([train_data.SibSp],train_data.Survived).style.background_gradient('summer_r')


# In[39]:


f, ax = plt.subplots(1, 2, figsize=(20, 8))

# Bar plot for SibSp vs Survived (without hue)
sns.barplot(x='SibSp', y='Survived', data=train_data, ax=ax[0])
ax[0].set_title('SibSp vs Survived in Bar Plot')

# Bar plot for SibSp vs Survived (with hue)
sns.barplot(x='SibSp', y='Survived', hue='Sex', data=train_data, ax=ax[1])
ax[1].set_title('SibSp vs Survived with hue')
plt.legend(title='Sex', loc='upper right')  # Add a legend for the hue

plt.show()


# In[40]:


cross_tab = pd.crosstab(train_data.SibSp, train_data.Pclass)

# Create a heatmap
plt.figure(figsize=(10, 6))
sb.heatmap(cross_tab, annot=True, fmt='d', cmap='summer_r')
plt.title('SibSp vs Pclass (Cross-tabulation)')
plt.xlabel('Pclass')
plt.ylabel('SibSp')
plt.show()


# In[ ]:




