
# coding: utf-8

# In[1]:


import collections
import glob
from itertools import chain
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')
np.random.seed(1)


# In[2]:


files = glob.glob('./res/writer_user_doc.txt')

words = []
for f in files:
    file = open(f)
    words.append(file.read())
    file.close()

words = list(chain.from_iterable(words))
words = ''.join(words)[:-1]
sentences = words.split('\n')


# In[3]:


sentences_df = pd.DataFrame(sentences)


# In[4]:


sentences_df['user'] = sentences_df[0].apply(lambda x : x.split()[0])
sentences_df['words'] = sentences_df[0].apply(lambda x : ' '.join(x.split()[1:]))


# In[5]:


sentences_df_indexed = sentences_df.reset_index().set_index('user')


# In[8]:


final_doc_embeddings = np.load('./doc_embeddings.npy')


# In[25]:


def most_similar(user_id, size):
    if user_id in sentences_df_indexed.index:
        user_index = sentences_df_indexed.loc[user_id]['index']
        dist = final_doc_embeddings.dot(final_doc_embeddings[user_index][:,None])
        closest_doc = np.argsort(dist,axis=0)[-size:][::-1]
        furthest_doc = np.argsort(dist,axis=0)[0][::-1]

        result = []
        for idx, item in enumerate(closest_doc):
            user = sentences[closest_doc[idx][0]].split()[0]
            dist_value = dist[item][0][0]
            result.append([user, dist_value])
        return result


# In[20]:


def similar(user_id, writer_id):
    if user_id in sentences_df_indexed.index and writer_id in sentences_df_indexed.index:
        user_index = sentences_df_indexed.loc[user_id]['index']
        writer_index = sentences_df_indexed.loc[writer_id]['index']
        dist = final_doc_embeddings[user_index].dot(final_doc_embeddings[writer_index])
        #print('{} - {} : {}'.format(user_id, writer_id, dist))
        return dist


# In[30]:


most_similar('#a0df5bd0e5a5bbc28b87f8c64462667c', 10)


# In[31]:


similar('#d6866a498157771069fdf15361cb012b', '@seochogirl')


# In[32]:


similar('#d6866a498157771069fdf15361cb012b', '@brunch')

