
# coding: utf-8

# In[76]:


import collections
import glob
from itertools import chain
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
np.random.seed(1)


# In[77]:


files = glob.glob('./res/writer_user_sentences.txt')

words = []
for f in files:
    file = open(f)
    words.append(file.read())
    file.close()

words = list(chain.from_iterable(words))
words = ''.join(words)[:-1]
sentences = words.split('\n')


# In[78]:


sentences_df = pd.DataFrame(sentences)


# In[79]:


sentences_df['user'] = sentences_df[0].apply(lambda x : x.split()[0])
sentences_df['words'] = sentences_df[0].apply(lambda x : ' '.join(x.split()[1:]))


# In[80]:


sentences_df.shape


# In[81]:


sentences_df_indexed = sentences_df.reset_index().set_index('user')


# In[136]:


final_doc_embeddings = np.load('./doc_embeddings_keyword_t20.npy')


# In[137]:


final_doc_embeddings


# In[138]:


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


# In[139]:


from scipy import spatial

def similar(user_id, writer_id):
    if user_id in sentences_df_indexed.index and writer_id in sentences_df_indexed.index:
        user_index = sentences_df_indexed.loc[user_id]['index']
        writer_index = sentences_df_indexed.loc[writer_id]['index']
        sim = spatial.distance.cosine(final_doc_embeddings[user_index], final_doc_embeddings[writer_index])
        print('{} - {} : {}'.format(user_id, writer_id, sim))
        return sim


# In[140]:


most_similar('#87a6479c91e4276374378f1d28eb307c', 10)


# In[141]:


similar('#d6866a498157771069fdf15361cb012b', '@seochogirl')
similar('#d6866a498157771069fdf15361cb012b', '@brunch')
similar('#87a6479c91e4276374378f1d28eb307c', '@begintalk')
similar('#87a6479c91e4276374378f1d28eb307c', '@tnrud572')
similar('#a0df5bd0e5a5bbc28b87f8c64462667c', '@kimmh12728xrf')
similar('#a0df5bd0e5a5bbc28b87f8c64462667c', '@brunch')
similar('#ec0fb734ba02a29c62c64e7ac7a8f13e', '@sethahn')
similar('#ec0fb734ba02a29c62c64e7ac7a8f13e', '@nomadesk')

