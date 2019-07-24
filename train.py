
# coding: utf-8

# In[1]:


import collections
import glob
from itertools import chain
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
np.random.seed(1)


# In[86]:


files = glob.glob('./res/writer_user_sentences_keyword.txt')

words = []
for f in files:
    file = open(f)
    words.append(file.read())
    file.close()

words = list(chain.from_iterable(words))
words = ''.join(words)[:-1]
sentences = words.split('\n')


# In[87]:


sentences_df = pd.DataFrame(sentences)


# In[88]:


sentences_df['user'] = sentences_df[0].apply(lambda x : x.split()[0])
sentences_df['words'] = sentences_df[0].apply(lambda x : ' '.join(x.split()[1:]))
sentences_df['words_list']  = sentences_df[0].apply(lambda x : x.split())
sentences_df['words_num'] = sentences_df[0].apply(lambda x : len(x.split()))


# In[89]:


len(set(sum(sentences_df.head(3000)['words_list'].tolist(), [])))


# In[90]:


sentences_df['words_num'].sum()


# In[39]:


sentences_df_indexed = sentences_df.reset_index().set_index('user')


# In[47]:


vocabulary_size = 400000

def build_dataset(sentences):
    words = ''.join(sentences).split()
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    
    unk_count = 0
    sent_data = []
    for sentence in sentences:
        data = []
        for word in sentence.split():
            if word in dictionary:
                index = dictionary[word]
            else:
                index = 0  # dictionary['UNK']
                unk_count = unk_count + 1
            data.append(index)
        sent_data.append(data)
    
    count[0][1] = unk_count
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys())) 
    return sent_data, count, dictionary, reverse_dictionary

data, count, dictionary, reverse_dictionary = build_dataset(sentences_df_indexed['words'].tolist())
print('Most common words (+UNK)', count[:5])
print('Sample data', data[:2])
# del words  # Hint to reduce memory.


# In[48]:


skip_window = 5
instances = 0

# Pad sentence with skip_windows
for i in range(len(data)):
    data[i] = [vocabulary_size]*skip_window+data[i]+[vocabulary_size]*skip_window

# Check how many training samples that we get    
for sentence  in data:
    instances += len(sentence)-2*skip_window
print(instances)


# In[49]:


context = np.zeros((instances,skip_window*2+1),dtype=np.int32)
labels = np.zeros((instances,1),dtype=np.int32)
doc = np.zeros((instances,1),dtype=np.int32)

k = 0
for doc_id, sentence  in enumerate(data):
    for i in range(skip_window, len(sentence)-skip_window):
        context[k] = sentence[i-skip_window:i+skip_window+1] # Get surrounding words
        labels[k] = sentence[i] # Get target variable
        doc[k] = doc_id
        k += 1
        
context = np.delete(context,skip_window,1) # delete the middle word        
        
shuffle_idx = np.random.permutation(k)
labels = labels[shuffle_idx]
doc = doc[shuffle_idx]
context = context[shuffle_idx]


# In[50]:


batch_size = 256
context_window = 2*skip_window
embedding_size = 50 # Dimension of the embedding vector.
softmax_width = embedding_size # +embedding_size2+embedding_size3
num_sampled = 5 # Number of negative examples to sample.
sum_ids = np.repeat(np.arange(batch_size),context_window)

len_docs = len(data)

graph = tf.Graph()

with graph.as_default(): # , tf.device('/cpu:0')
    # Input data.
    train_word_dataset = tf.placeholder(tf.int32, shape=[batch_size*context_window])
    train_doc_dataset = tf.placeholder(tf.int32, shape=[batch_size])
    train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])

    segment_ids = tf.constant(sum_ids, dtype=tf.int32)

    word_embeddings = tf.Variable(tf.random_uniform([vocabulary_size,embedding_size],-1.0,1.0))
    word_embeddings = tf.concat([word_embeddings,tf.zeros((1,embedding_size))],0)
    doc_embeddings = tf.Variable(tf.random_uniform([len_docs,embedding_size],-1.0,1.0))

    softmax_weights = tf.Variable(tf.truncated_normal([vocabulary_size, softmax_width],
                             stddev=1.0 / np.sqrt(embedding_size)))
    softmax_biases = tf.Variable(tf.zeros([vocabulary_size]))

    # Model.
    # Look up embeddings for inputs.
    embed_words = tf.segment_mean(tf.nn.embedding_lookup(word_embeddings, train_word_dataset),segment_ids)
    embed_docs = tf.nn.embedding_lookup(doc_embeddings, train_doc_dataset)
    embed = (embed_words+embed_docs)/2.0#+embed_hash+embed_users

    # Compute the softmax loss, using a sample of the negative labels each time.
    loss = tf.reduce_mean(tf.nn.nce_loss(softmax_weights, softmax_biases, train_labels, 
                                         embed, num_sampled, vocabulary_size))

    # Optimizer.
    optimizer = tf.train.AdagradOptimizer(0.5).minimize(loss)
        
    norm = tf.sqrt(tf.reduce_sum(tf.square(doc_embeddings), 1, keep_dims=True))
    normalized_doc_embeddings = doc_embeddings / norm


# In[51]:



############################
# Chunk the data to be passed into the tensorflow Model
###########################
data_idx = 0
def generate_batch(batch_size):
    global data_idx

    if data_idx+batch_size<instances:
        batch_labels = labels[data_idx:data_idx+batch_size]
        batch_doc_data = doc[data_idx:data_idx+batch_size]
        batch_word_data = context[data_idx:data_idx+batch_size]
        data_idx += batch_size
    else:
        overlay = batch_size - (instances-data_idx)
        batch_labels = np.vstack([labels[data_idx:instances],labels[:overlay]])
        batch_doc_data = np.vstack([doc[data_idx:instances],doc[:overlay]])
        batch_word_data = np.vstack([context[data_idx:instances],context[:overlay]])
        data_idx = overlay
    batch_word_data = np.reshape(batch_word_data,(-1,1))

    return batch_labels, batch_word_data, batch_doc_data


# In[52]:


num_steps = 200001
step_delta = int(num_steps/20)


with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()
    print('Initialized')
    average_loss = 0
    for step in range(num_steps):
        batch_labels, batch_word_data, batch_doc_data        = generate_batch(batch_size)
        feed_dict = {train_word_dataset : np.squeeze(batch_word_data),
                     train_doc_dataset : np.squeeze(batch_doc_data),
                     train_labels : batch_labels}
        _, l = session.run([optimizer, loss], feed_dict=feed_dict)
        average_loss += l
        if step % step_delta == 0:
            if step > 0:
                average_loss = average_loss / step_delta
            # The average loss is an estimate of the loss over the last 2000 batches.
            print('Average loss at step %d: %f' % (step, average_loss))
            average_loss = 0
    save_path = tf.train.Saver().save(session, "./model/doc2vec_model")    
    # restore model
    #tf.train.Saver().restore(session, "./model/doc2vec_model")  
    
    # Get the weights to save for later
    final_word_embeddings = word_embeddings.eval()
    final_word_embeddings_out = softmax_weights.eval()
    final_doc_embeddings = normalized_doc_embeddings.eval()


# In[75]:


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


# In[76]:


most_similar('#a0df5bd0e5a5bbc28b87f8c64462667c', 20)


# In[77]:


most_similar('#87a6479c91e4276374378f1d28eb307c', 10)


# In[78]:


from scipy import spatial

def similar(user_id, writer_id):
    if user_id in sentences_df_indexed.index and writer_id in sentences_df_indexed.index:
        user_index = sentences_df_indexed.loc[user_id]['index']
        writer_index = sentences_df_indexed.loc[writer_id]['index']
        dist= spatial.distance.cosine(final_doc_embeddings[user_index], final_doc_embeddings[writer_index])
        print('{} - {} : {}'.format(user_id, writer_id, dist))
        return dist


# In[82]:


np.save('doc_embeddings_keyword', final_doc_embeddings)


# In[81]:


similar('#d6866a498157771069fdf15361cb012b', '@seochogirl')
similar('#d6866a498157771069fdf15361cb012b', '@brunch')
similar('#87a6479c91e4276374378f1d28eb307c', '@begintalk')
similar('#87a6479c91e4276374378f1d28eb307c', '@tnrud572')
similar('#a0df5bd0e5a5bbc28b87f8c64462667c', '@kimmh12728xrf')
similar('#a0df5bd0e5a5bbc28b87f8c64462667c', '@brunch')
similar('#ec0fb734ba02a29c62c64e7ac7a8f13e', '@sethahn')
similar('#ec0fb734ba02a29c62c64e7ac7a8f13e', '@nomadesk')

