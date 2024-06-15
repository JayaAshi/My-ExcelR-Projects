import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import matplotlib.pyplot as plt
import pickle


true_df = pd.read_csv(r"C:\Users\Shiva\PycharmProjects\working nlp\True.csv",encoding='Latin-1',on_bad_lines='skip',low_memory=False)
true_df['status']=true_df.apply(lambda x: 'true',axis=1)

false_df =pd.read_csv(r"C:\Users\Shiva\PycharmProjects\working nlp\Fake.csv",encoding='Latin-1',on_bad_lines='skip',low_memory=False)
false_df['status']=false_df.apply(lambda x: 'false',axis=1)

data = pd.concat([true_df,false_df], axis=0)
data_subject=data
data.head(100)
data =data.drop(['subject','date'],axis=1)


data['total']=data['title']+' '+data['text']

data.drop_duplicates(inplace=True)
data.dropna(inplace=True)

from sklearn.model_selection import train_test_split

x = data['total']  # this time we want to look at the text
y = data['status']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

# Convert text data to numerical features using count vectorization

LR = Pipeline([('CV', CountVectorizer()),
                     ('clf', LogisticRegression()),
])

# Feed the training data through the pipeline
LR.fit(x_train,y_train)

pickle.dump(LR,open('model1.pkl','wb'))
