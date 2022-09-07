#!/usr/bin/env python
# coding: utf-8

# In[6]:


pip install pyodbc


# In[7]:


import pyodbc 
conn = pyodbc.connect('Driver={SQL Server};'
                      'Server=LAPTOP-LDK3555G\SQLEXPRESS2017;'
                      'Database=test;'
                      'Trusted_Connection=yes;')


# In[8]:


sql_query = pd.read_sql_query(''' 
                              select * from test.dbo.Labtest
                              '''
                              ,conn)


# In[9]:


df = pd.DataFrame(sql_query)


# In[12]:


import pandas as pd
import neattext.functions as nfx
# Load ML/Rc Pkgs
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity,linear_kernel


# In[11]:


pip install neattext


# In[13]:


df.head()


# In[15]:


df['experiment']


# In[17]:


df['clean_experiment'] = df['experiment'].apply(nfx.remove_stopwords)
# Clean Text:stopwords,special charac
df['clean_experiment'] = df['clean_experiment'].apply(nfx.remove_special_characters)
df[['experiment','clean_experiment']]


# In[18]:


# Vectorize our Text
count_vect = CountVectorizer()
cv_mat = count_vect.fit_transform(df['clean_experiment'])
# Sparse
cv_mat


# In[19]:


# Dense
cv_mat.todense()


# In[20]:


df_cv_words = pd.DataFrame(cv_mat.todense(),columns=count_vect.get_feature_names())
df_cv_words.head()


# In[21]:


# Cosine Similarity Matrix
cosine_sim_mat = cosine_similarity(cv_mat)
cosine_sim_mat


# In[22]:


# import seaborn as sns
# sns.heatmap(cosine_sim_mat[0:10],annot=True)
df.head()


# In[25]:


# Get Course ID/Index
experiment_indices = pd.Series(df.index,index=df['experiment']).drop_duplicates()
experiment_indices


# In[30]:


idx=experiment_indices['DISPERSIVE POWER OF THE MATERIAL OF A\r\nPRISM – SPECTROMETER']


# In[31]:


scores = list(enumerate(cosine_sim_mat[idx]))
scores


# In[32]:


# Sort our scores per cosine score
sorted_scores = sorted(scores,key=lambda x:x[1],reverse=True)
# Omit the First Value/itself
sorted_scores[1:]


# In[33]:


# Selected Courses Indices
selected_experiment_indices = [i[0] for i in sorted_scores[1:]]
selected_experiment_indices


# In[35]:


# Selected Courses Scores
selected_experiment_scores = [i[1] for i in sorted_scores[1:]]
recommended_result = df['experiment'].iloc[selected_experiment_indices]
rec_df = pd.DataFrame(recommended_result)
rec_df.head()


# In[36]:


rec_df['similarity_scores'] = selected_experiment_scores
rec_df


# In[42]:


def recommend_experiment(experiment,num_of_rec=10):
    # ID for title
    idx = experiment_indices[experiment]
    # Course Indice
    # Search inside cosine_sim_mat
    scores = list(enumerate(cosine_sim_mat[idx]))
    # Scores
    # Sort Scores
    sorted_scores = sorted(scores,key=lambda x:x[1],reverse=True)
    # Recomm
    selected_experiment_indices = [i[0] for i in sorted_scores[1:]]
    selected_experiment_scores = [i[1] for i in sorted_scores[1:]]
    result = df['experiment'].iloc[selected_experiment_indices]
    rec_df = pd.DataFrame(result)
    rec_df['similarity_scores'] = selected_experiment_scores
    return rec_df.head(num_of_rec) 


# In[45]:


recommend_experiment(' Estimation of ferrous iron in cement by colorimetric method',5)


# In[ ]:





# In[ ]:




