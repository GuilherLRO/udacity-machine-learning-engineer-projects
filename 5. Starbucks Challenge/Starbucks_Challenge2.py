#!/usr/bin/env python
# coding: utf-8

# In[4]:


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
from xgboost import XGBClassifier

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix


# ### Your task is to combine transaction, demographic and offer data to determine which demographic groups respond best to which offer type. This data set is a simplified version of the real Starbucks app because the underlying simulator only has one product whereas Starbucks actually sells dozens of products.

# # 01. Data gathering and cleaning 

# In[5]:


# list all files in the StarterFiles directory
print(os.listdir('StarterFiles'))


# ## 1.1. Portfolio
# - id (string) - offer id
# - offer_type (string) - type of offer ie BOGO, discount, informational
# - difficulty (int) - minimum required spend to complete an offer
# - reward (int) - reward given for completing an offer
# - duration (int) - time for offer to be open, in days
# - channels (list of strings)

# In[6]:


# reading 'portfolio.json' file into a pandas dataframe
portfolio_ = pd.read_json('StarterFiles/portfolio.json'  , orient='records', lines=True)
portfolio_.head(3)


# In[7]:


portfolio_.info()


# In[8]:


# Data preprocessing

# one-hot encoding channels column
portfolio = portfolio_.copy()
portfolio['channels'] = portfolio['channels'].apply(lambda x: ','.join(map(str, x)))
portfolio = portfolio.join(portfolio['channels'].str.get_dummies(','))
portfolio.drop('channels', axis=1, inplace=True)

# one-hot encoding offer_type column
portfolio = portfolio.join(pd.get_dummies(portfolio['offer_type'])) 
portfolio.drop('offer_type', axis=1, inplace=True)

#drop email column since it it contains no useful information
portfolio.drop('email', axis=1, inplace=True)

#rename id column to offer_id, reward to offer_reward and duration to offer_duration
portfolio.rename(columns={'id':'offer_id', 'reward':'offer_reward', 'duration':'offer_duration'}, inplace=True)
portfolio


# ## 1.2. Profile
# - age (int) - age of the customer
# - became_member_on (int) - date when customer created an app account
# - gender (str) - gender of the customer (note some entries contain 'O' for other rather than M or F)
# - id (str) - customer id
# - income (float) - customer's income

# In[9]:


# reading 'profile.json' file into a pandas dataframe
profile_ = pd.read_json('StarterFiles/profile.json'      , orient='records', lines=True)
profile_.head(3)


# In[10]:


profile_.info()


# In[11]:


# Creating a figure with three subplots
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Plotting the distribution of gender column
profile_.fillna('NA').groupby('gender').count()['id'].plot(kind='bar', color='#0fa7d1', rot=0, ax=axes[0])
axes[0].set_title("Gender Distribution")
axes[0].set_xlabel("Gender")

# Plotting the age distribution of customers
profile_.age.hist(bins=40, color='#0fa7d1', grid=False, edgecolor='white', ax=axes[1])
axes[1].set_title("Age Distribution")
axes[1].set_xlabel("Age")

# Plotting the income distribution of customers
profile_.income.hist(bins=40, color='#0fa7d1', grid=False, edgecolor='white', ax=axes[2])
axes[2].set_title("Income Distribution")
axes[2].set_xlabel("Income")

# Adjusting spacing between subplots
plt.tight_layout()

# Displaying the figure
plt.show()


# In[12]:


# Since there is incomplete data over the gender and some a sinilar amout of age outliers, let's veryfy if they are the same customers
df_temp_outliers = profile_[profile_.isnull().any(axis=1)].fillna('NA')
pd.crosstab(df_temp_outliers['age'], df_temp_outliers['gender'])


# In[13]:


# Data preprocessing

profile = profile_.copy()
# Since the age outliers and missing age values are the same customers, we lets identify them

# identify the age outliers
profile['imcomplete_data'] = profile['gender'].isnull()

#rename id column to customer_id, became_member_on to became_member_date and income to customer_income
profile.rename(columns={'id':'customer_id', 'became_member_on':'became_member_date', 'income':'customer_income'}, inplace=True)

# adjust became_member_date from string in format YYYYMMDD to datetime
profile['became_member_date'] = pd.to_datetime(profile['became_member_date'], format='%Y%m%d')

# create a new column with the number of days since the customer became a member
profile['memberdays'] = (datetime.datetime.today() - profile['became_member_date']).dt.days

# one hot encoding gender column
profile['gender_M'] = profile['gender'].apply(lambda x: 1 if x == 'M' else 0)
profile['gender_F'] = profile['gender'].apply(lambda x: 1 if x == 'F' else 0)
profile['gender_0'] = profile['gender'].apply(lambda x: 1 if x == '0' else 0)


profile.head(3)


# ## 1.3. Transcript 
# - event (str) - record description (ie transaction, offer received, offer viewed, etc.)
# - person (str) - customer id
# - time (int) - time in hours since start of test. The data begins at time t=0
# - value - (dict of strings) - either an offer id or transaction amount depending on the record

# In[14]:


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime


# In[15]:


# reading 'transcript.json' file into a pandas dataframe
transcript_ = pd.read_json('StarterFiles/transcript.json', orient='records', lines=True)
transcript_.head(5)


# In[16]:


transcript_.info()


# In[17]:


transcript_.query('event == "offer completed"').head(3)


# In[18]:


# quantity of events by type
transcript_.groupby('event').count()['person'].plot(kind = 'bar', figsize = (6, 3), color = '#0fa7d1', rot = 0)
plt.title('Quantity of events by type')


# In[19]:


# Data preprocessing
transcript = transcript_.copy()
#rename person column to customer_id
transcript.rename(columns={'person':'customer_id'}, inplace=True)

transcript['offer_id'] = transcript['value'].apply(lambda x: x.get('offer id')  if x.get('offer id') != None else x.get('offer_id'))
transcript['amount'] = transcript['value'].apply(lambda x: x.get('amount'))
# transcript.drop('value', axis=1, inplace=True)

# one-hot encoding event column with a for loop
for event in transcript['event'].drop_duplicates().reset_index(drop=True):
    try:
        transcript[event.split(' ')[1]] = transcript['event'].apply(lambda x: 1 if x == event else 0)
    except:
        transcript[event.split(' ')[0]] = transcript['event'].apply(lambda x: 1 if x == event else 0)

transcript.query('event == "offer completed"').head(3)


# In[20]:


# https://towardsdatascience.com/starbucks-capstone-challenge-35e3b8c6b328
# https://medium.com/swlh/starbucks-capstone-challenge-350575a03f9a


# In[21]:


#plotting the distribution of the amount column


# In[22]:


# 02. Buiding offer dataset


# In[23]:


transcript[['completed', 'received', 'viewed']].sum().plot(kind='bar', figsize=(6, 3), color='#0fa7d1', rot=0)


# In[24]:


offers = transcript.copy().groupby(['customer_id', 'offer_id']).sum()[[ 'viewed',  'received', 'completed', 'transaction']].reset_index().reset_index().sample(frac=1).reset_index(drop=True)
offers['percent_completed'] = offers['completed'] / offers['received']


# In[25]:


offers.query('completed <= received').count() / offers.count()


# In[26]:


#joined df
df_ = offers.merge(portfolio, on='offer_id', how='inner').merge(profile, on='customer_id', how='inner')

df_['offer_completed'] = df_['percent_completed'].apply(lambda x: 1 if x > 0.5 else 0)
df_['offer_recieved']  = df_['received'].apply(lambda x: 1 if x > 0 else 0)
df_['transaction_made'] = df_['transaction'].apply(lambda x: 1 if x > 0 else 0)

df_ =df_.query('offer_recieved == 1')

df_ = df_[df_['imcomplete_data'] == False]
df_.drop(['imcomplete_data', 'became_member_date','gender'], axis=1, inplace=True)
df_.drop(['gender_0'], axis=1, inplace=True)
df_.drop(['customer_id', 'offer_id'], axis=1, inplace=True)
df_.drop(['index','viewed','received','completed','transaction','percent_completed','transaction_made','offer_recieved','gender_F'], axis=1, inplace=True)

df_.shape


# In[27]:


df_


# In[28]:


df_[['offer_completed']].hist()


# In[29]:


# dividing the dataset train, validation and test
from sklearn.model_selection import train_test_split

X = df_.drop('offer_completed', axis=1)
y = df_['offer_completed']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=df_['offer_completed'])
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)


# In[30]:


# 03. Building the model
# creating a function to train and evaluate the model
def train_evaluate_model(model, X_train, y_train, X_val, y_val):
    # training the model
    model.fit(X_train, y_train)
    
    # predicting the model
    y_pred = model.predict(X_val)

    # evaluating the model
    accuracy = accuracy_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred)
    recall = recall_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    roc_auc = roc_auc_score(y_val, y_pred)

    # printing the metrics
    print('Accuracy: {:.2f}%'.format(accuracy*100))
    print('Precision: {:.2f}%'.format(precision*100))
    print('Recall: {:.2f}%'.format(recall*100))
    print('F1 Score: {:.2f}%'.format(f1*100))
    print('ROC AUC: {:.2f}%'.format(roc_auc*100))

    # plotting the confusion matrix and ROC AUC curve side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 3))

    # Confusion Matrix
    cm = confusion_matrix(y_val, y_pred)
    sns.heatmap(cm, annot=True, ax=ax1, cmap='Blues', fmt='g')
    ax1.set_xlabel('Predicted labels')
    ax1.set_ylabel('True labels')
    ax1.set_title('Confusion Matrix')
    ax1.xaxis.set_ticklabels(['No', 'Yes'])
    ax1.yaxis.set_ticklabels(['No', 'Yes'])

    # ROC Curve
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_val, y_pred_proba)
    ax2.plot([0, 1], [0, 1], 'k--')
    ax2.plot(fpr, tpr)
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')
    ax2.set_title('ROC Curve')

    # Adjust the spacing between plots
    plt.tight_layout()

    # Show the plots
    plt.show()


# In[31]:


# 03.1. XGBoost Classifier
# training and evaluating the model
xgb = XGBClassifier(random_state=42)
train_evaluate_model(xgb, X_train, y_train, X_val, y_val)


# In[32]:


# most important features plot
xgb.fit(X_train, y_train)
feature_importances = pd.DataFrame(xgb.feature_importances_, index=X_train.columns, columns=['importance']).sort_values('importance', ascending=False)
feature_importances.plot(kind='barh', figsize=(6, 3), color='#0fa7d1', rot=0)
plt.xlabel('Importance')
plt.ylabel('Features')
plt.title('Feature Importance')


# In[33]:


# import classification_report
from sklearn.metrics import classification_report
predictions = xgb.predict(X_test)
print(classification_report(y_test, predictions))


# In[36]:


data = X_test.iloc[0]

# data to json
data_json = data.to_json()
data_json


# In[37]:


X_test.columns


# In[39]:


df_['predictions'] = xgb.predict(X)


# In[41]:


df_.query('predictions == 1')


# In[ ]:


import json
# Hardcoded data for prediction
hardcoded_data = {
"offer_reward":30.0,
"difficulty":30.0,
"offer_duration":30.0,
"mobile":1.0,
"social":1.0,
"web":1.0,
"bogo":1.0,
"discount":10.0,
"informational":1.0,
"age":58.0,
"customer_income":550000.0,
"memberdays":32981.0,
"gender_M":1.0
}

# Convert the hardcoded data to a list of values
input_data = [list(hardcoded_data.values())]

# Create a low-level SageMaker client
sagemaker_client = boto3.client('sagemaker-runtime')

# Specify the endpoint name
endpoint_name = 'xgboost-endpoint'

# Convert the input data to CSV format
input_data_csv = ','.join([str(val) for val in input_data[0]])

# Make a request to the endpoint
response = sagemaker_client.invoke_endpoint(
    EndpointName=endpoint_name,
    ContentType='text/csv',
    Body=input_data_csv
)

# Parse the response
predictions = response['Body'].read().decode()
predictions = json.loads(predictions)

if predictions[0]['predicted_label'] == 1:
    print('The offer will be completed. (probability: {:.2f}%)'.format(predictions[0]['probability']*100))
else:
    print('The offer will not be completed. (probability: {:.2f}%)'.format(predictions[0]['probability']*100))
    

