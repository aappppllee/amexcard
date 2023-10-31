#!/usr/bin/env python
# coding: utf-8

# In[9]:


get_ipython().system('pip install tensorflow')
get_ipython().system('pip install sklearn')


# In[2]:


from tensorflow.keras.layers import Embedding
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


# In[3]:


def train_model_on_chunk(chunk, scaler,imputer):
    chunk['result'] = chunk.apply(map_to_result,axis=1)
    columns_to_drop = ['ind_recommended', 'activation','merchant','customer']
    chunk.drop(columns_to_drop, axis=1, inplace=True)
    X_chunk = chunk.drop('result',axis=1)
    X_chunk = pd.DataFrame(imputer.fit_transform(X_chunk), columns=X_chunk.columns)
    X_chunk['merchant_profile_01'] = X_chunk['merchant_profile_01'].astype(int)
    num_categories = len(X_chunk['merchant_profile_01'].unique())
    embedding_dim = 32
    
    X_chunk = X_chunk.values
    y_chunk = chunk['result'].values
    
    X_chunk = scaler.fit_transform(X_chunk)
    
    print("ok")
    
    # Encode target variable into one-hot vectors
    y_chunk = keras.utils.to_categorical(y_chunk - 1, num_classes=4)  # Assuming classes are 1, 2, 3, 4
    
    print("ok")
    # Initialize the model
    model = keras.Sequential([
        keras.layers.Input(shape=(67,)),  # Input layer with 69 features
        keras.layers.Dense(128, activation='relu'),  # Hidden layer with 128 neurons and ReLU activation   
        keras.layers.Dense(64, activation='relu'),   # Hidden layer with 64 neurons and ReLU activation  
        keras.layers.Dense(4, activation='softmax')  # Output layer with 4 neurons for multiclass classification
    ])
    
    
    print("ok")
    
    # Add an embedding layer to your model
    #model.add(Embedding(input_dim=num_categories, output_dim=embedding_dim, input_length=1))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    print("ok")
    
    # Train the model on the current chunk
    model.fit(X_chunk, y_chunk, epochs=20, batch_size=2000, validation_split=0.2)
    
    return model


# In[4]:


def map_to_result(row):
    ind_recommended = row['ind_recommended']  # Assuming 'ind_recommended' is the first column
    activation = row['activation']
    if ind_recommended == 1 and activation == 1:
        return 1
    elif ind_recommended == 0 and activation == 0:
        return 2
    elif ind_recommended == 1 and activation == 0:
        return 3
    elif ind_recommended == 0 and activation == 1:
        return 4


# In[8]:


#gcs_bucket = "gs://american_express/training_data.csv"
file_path = "gs://american_express/lastfile.csv"

batch_size = 200000


chunks = pd.read_csv(diverse,chunksize=batch_size)


# In[ ]:





# In[9]:


models=[]


# In[10]:


scaler = StandardScaler()
imputer = SimpleImputer(strategy='most_frequent')


# In[11]:


for chunk in chunks:
    #print(chunk.shape)
    #add = addition.sample(n=100000)
    #print(add.shape)
    #df = pd.concat([chunk,add],axis=0)
    #df.reset_index(drop=True, inplace=True)
    model = train_model_on_chunk(chunk, scaler,imputer)
    models.append(model)
    print(chunk.shape)


# In[62]:


eval_file = "gs://american_express/eval_data.csv"
batch_size = 100000


chunks = pd.read_csv(eval_file,chunksize=batch_size)


# In[63]:


desired_order = ['customer_digital_activity_04',
       'customer_spend_01', 'customer_industry_spend_01',
       'customer_industry_spend_02', 'customer_industry_spend_03',
       'customer_industry_spend_04', 'customer_industry_spend_05',
       'customer_spend_02', 'customer_spend_03', 'customer_merchant_02',
       'customer_merchant_01', 'customer_spend_04', 'customer_spend_05',
       'customer_spend_06', 'customer_spend_07', 'merchant_spend_01',
       'merchant_spend_02', 'merchant_spend_03', 'merchant_spend_04',
       'merchant_spend_05', 'merchant_spend_06', 'merchant_spend_07',
       'merchant_spend_08', 'merchant_profile_01', 'customer_merchant_03',
       'customer_profile_01', 'customer_profile_02',
       'customer_digital_activity_05', 'customer_spend_13',
       'customer_digital_activity_06', 'customer_spend_14',
       'customer_digital_activity_07', 'customer_digital_activity_08',
       'customer_digital_activity_09', 'customer_digital_activity_10',
       'customer_digital_activity_11', 'customer_digital_activity_12',
       'customer_digital_activity_13', 'customer_digital_activity_14',
       'customer_digital_activity_15', 'customer_spend_15',
       'customer_digital_activity_16', 'customer_spend_16',
       'customer_spend_17', 'customer_digital_activity_17',
       'customer_digital_activity_03', 'merchant_spend_11',
       'customer_digital_activity_18', 'customer_digital_activity_19',
       'distance_01', 'customer_digital_activity_20', 'distance_02',
       'distance_03', 'customer_spend_18', 'customer_spend_19',
       'customer_digital_activity_21', 'customer_digital_activity_22',
       'distance_04', 'merchant_profile_02', 'merchant_spend_09',
       'merchant_profile_03', 'customer_digital_activity_01',
       'merchant_spend_10', 'customer_profile_03',
       'customer_digital_activity_02', 'customer_profile_04', 'distance_05',
       'customer', 'merchant']


# In[64]:


output_file_path = f'gs://american_express/output_{count}.csv'


# In[65]:


count = 0


# In[66]:


for chunk in chunks:
    chunk = chunk[desired_order]
    df = chunk.copy()
    columns_to_drop = ['merchant','customer']
    chunk.drop(columns_to_drop, axis=1, inplace=True)
    X_chunk = pd.DataFrame(imputer.fit_transform(chunk), columns=chunk.columns)
    X_chunk['merchant_profile_01'] = X_chunk['merchant_profile_01'].astype(int)
    X_chunk = X_chunk.values
    X_chunk = scaler.fit_transform(X_chunk)
    predictions = [model.predict(X_chunk) for model in models]
    class_1_probs = np.mean([pred[:, 0] for pred in predictions], axis=0)
    class_2_probs = np.mean([pred[:, 3] for pred in predictions], axis=0)
    prediction = class_1_probs / (class_1_probs + class_2_probs)
    df['predicted_score'] = prediction
    df = df[['customer','merchant','predicted_score']]
    df.to_csv(f"gs://american_express/output_{count}.csv", index=False, header=False, mode='a')
    count+=1
print(count)


# In[ ]:




