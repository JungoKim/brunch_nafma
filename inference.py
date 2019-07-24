#-*- encoding: utf-8 -*-
# inference using statistics, doc2vec
# mf removed after evaluation
# word2vec removed due to gensim license
# doc2vec removed and changed from gensim to tensorflow

import os, sys 
import pdb 
import tqdm
import json
import datetime
from util import iterate_data_files
import glob
from itertools import chain
import numpy as np
import pandas as pd
from scipy import spatial

np.random.seed(1)


# recommend a popular sequential articles thas was read in the dup dates
def find_dup_seq(viewer):
  recommends1 = []
  recommends2 = []
  if viewer in t_followings:
    followings = t_followings[viewer]
  else:
    followings = []

  if viewer in t_reads_dup:
    reads_org = t_reads_dup[viewer]
    reads = sorted(set(reads_org), key=lambda x: reads_org.index(x))  # dedup and keep order
    reads.reverse()   # the later the better
    num_reads = len(reads)
    for read in reads:
      if read in seq_read:
        seqs = seq_read[read]
        for seq in seqs:
          if seq not in t_reads[viewer]:
            if (seq not in recommends1) and (seq not in recommends2):
              writer = seq.split("_")[0]
              if writer in followings:
                recommends1.append(seq)
              else:
                recommends2.append(seq)
              break
          #if num_reads > 100: break
          if num_reads > 50: break  # for ent

      if num_reads < 50:
        if read in prev_read:
          seqs = prev_read[read]
          for seq in seqs:
            if seq not in t_reads[viewer]:
              if (seq not in recommends1) and (seq not in recommends2):
                writer = seq.split("_")[0]
                if writer in followings:
                  recommends1.append(seq)
                else:
                  recommends2.append(seq)
                break
  return recommends1, recommends2


# add new articles of following writer in reverse order (recent first)
# need to experiment to change windows and order for the case over 100
def find_new_articles(viewer):
  recommends1 = []
  recommends2 = []

  dup_read_writers = {}
  if viewer in t_reads_dup:
    reads = t_reads_dup[viewer]
    for read in reads:
      writer = read.split("_")[0]
      if writer not in dup_read_writers:
        dup_read_writers[writer] = 1

  read_writers = {}
  if viewer in t_reads:
    reads = t_reads[viewer]
    for read in reads:
      writer = read.split("_")[0]
      if writer not in read_writers:
        read_writers[writer] = 1

  if viewer in t_followings:
    followings = t_followings[viewer]
  else:
    followings = []
  if viewer in t_non_follow:
    non_follow = t_non_follow[viewer]
  else:
    non_follow = []

  if len(followings) == 0:
    if len(non_follow) == 0:
      #print("no followings no freq for", viewer)
      return recommends1, recommends2

  # sort by stats
  followings_sorted_stats = []  # 1st priority
  if viewer in t_reads:
    followings_cnt = {}
    all_reads = t_reads[viewer]
    for read in all_reads:
      writer = read.split("_")[0]
      if writer in followings:
        if writer in followings_cnt:
          followings_cnt[writer] += 1
        else:
          followings_cnt[writer] = 1
    followings_cnt_sorted = sorted(followings_cnt.items(), key=lambda kv: kv[1], reverse=True)

    for writer, cnt in followings_cnt_sorted:
      if writer in followings:
        followings_sorted_stats.append(writer)

  # sort by d2v
  followings_sorted = []  # 2nd priority
  if viewer in sentences_df_indexed.index:
    followings_sim = []
    sims = {}
    for writer in followings:
      if writer in sentences_df_indexed.index:
        sim = similarity(viewer, writer)
        if sim not in sims:
          sims[sim] = 1
        else:
          sim -= 0.000001
          if sim not in sims:
            sims[sim] = 1
          else:
            sim -= 0.000001
            if sim not in sims:
              sims[sim] = 1

        followings_sim.append([writer, sim])
    followings_sim_sorted = sorted(followings_sim, key=lambda x:x[1], reverse=True)
    for item in followings_sim_sorted:
      followings_sorted.append(item[0])

    for writer in followings_sorted:
      if writer not in followings_sorted_stats:
        followings_sorted_stats.append(writer)

  for writer in followings:
    if writer not in followings_sorted_stats:
      followings_sorted_stats.append(writer)

  followings = followings_sorted_stats

  if len(non_follow) > 0:
    if len(followings) < 10:
      followings += non_follow[:2]  # for ent

  if viewer in t_reads:
    reads = t_reads[viewer]
  else:
    reads = []
    #print("no previous reads for", viewer)

  for writer in followings:
    if writer not in writer_articles: continue
    articles = writer_articles[writer]
    articles_sorted = sorted(articles, key=lambda x: x[1], reverse=False)
    for article, reg_datetime in articles_sorted:
      if reg_datetime <= "20190221000000": continue  # smaller window will make higher ent
      if reg_datetime >= "20190315000000": break
      if article in reads:
        #print("found article already read")
        continue
      if article not in recommends1 and article not in recommends2:
        if writer in dup_read_writers:
          recommends1.append(article)
        #else:
        elif writer != "@brunch" or writer in read_writers: # for higher ent
          recommends2.append(article)

  # order should be changed for ndcg reason
  if len(recommends1) > 70:
    recommends1 = []
    recommends2 = []
    for writer in followings:
      if writer not in writer_articles: continue
      articles = writer_articles[writer]
      articles_sorted = sorted(articles, key=lambda x: x[1], reverse=False)
      for article, reg_datetime in articles_sorted:
        if reg_datetime <= "20190301000000": continue  # smaller window will make higher ent
        if reg_datetime >= "20190313000000": break
        if article in reads:
          #print("found article already read")
          continue
        if article not in recommends1 and article not in recommends2:
          if writer in dup_read_writers:
            recommends1.append(article)
          #else:
          elif writer != "@brunch" or writer in read_writers: # for higher ent
            recommends2.append(article)

  return recommends1, recommends2


def read_test_user():
  print("read test user set", user_file)
  with open(user_file, "r") as fp:
    for line in fp:
      viewer_id = line.strip()
      t_users[viewer_id] = 1


def read_followings():
  print("read viewer followings")
  with open("res/users.json", "r") as fp:
    for line in fp:
      viewer = json.loads(line)
      if viewer['id'] in t_users:
        t_followings[viewer['id']] = viewer['following_list']
        if len(viewer['keyword_list']) > 0:
          t_keywords[viewer['id']] = []
          for keyword in viewer['keyword_list']:
            t_keywords[viewer['id']].append(keyword['keyword'])


# may need to write pickle for this
def read_reads():
  print("read reads of all users")
  files = sorted([path for path, _ in iterate_data_files('2018100100', '2019030100')])
  for path in tqdm.tqdm(files, mininterval=1):
    date = path[11:19]
    for line in open(path):
      tokens = line.strip().split()
      user_id = tokens[0]
      reads = tokens[1:]
      if len(reads) < 1: continue
      if user_id in t_users:
        if user_id in t_reads:
          t_reads[user_id] += reads
        else:
          t_reads[user_id] = reads
        if date >= "20190222":
          if user_id in t_reads_dup:
            t_reads_dup[user_id] += reads
          else:
            t_reads_dup[user_id] = reads

        reads_set = set(reads)
        for read in reads_set:
          writer = read.split("_")[0]
          if (user_id not in t_followings) or (writer not in t_followings[user_id]):
            if user_id in t_non_follows:
              if writer in t_non_follows[user_id]:
                t_non_follows[user_id][writer] += 1
              else:
                t_non_follows[user_id][writer] = 1
            else:
              t_non_follows[user_id] = {}
              t_non_follows[user_id][writer] = 1

      num_reads_n1 = len(reads)-1
      for i, read in enumerate(reads):
        if i < num_reads_n1:
          if read == reads[i+1]: continue   # when two continous reads are the same
          if read in seq_reads:
            if reads[i+1] in seq_reads[read]:
              seq_reads[read][reads[i+1]] += 1
            else:
              seq_reads[read][reads[i+1]] = 1
          else:
            seq_reads[read] = {}
            seq_reads[read][reads[i+1]] = 1

      for i, read in enumerate(reads):
        if i < num_reads_n1:
          nread = reads[i+1]
          if read == nread: continue   # when two continous reads are the same
          if nread in prev_reads:
            if read in prev_reads[nread]:
              prev_reads[nread][read] += 1
            else:
              prev_reads[nread][read] = 1
          else:
            prev_reads[nread] = {}
            prev_reads[nread][read] = 1

  for user in t_reads:
    if user not in t_reads_dup:
      t_reads_dup[user] = t_reads[user][-10:]


def determine_non_follow():
  print("find not following but favorite writers")
  for user in t_non_follows:
    writers = t_non_follows[user]
    writers_sorted = sorted(writers.items(), key=lambda x: x[1], reverse=True)
    if len(writers_sorted) < 3: tops = len(writers_sorted)
    else: tops = 3
    if writers_sorted[0][1] < 5: continue
    t_non_follow[user] = []
    for i in range(tops):
      if writers_sorted[i][1] < 5: break
      t_non_follow[user].append(writers_sorted[i][0])


# may need to write pickle for this
def determine_seq_read():
  print("find co-occurence of articles")
  for article in seq_reads:
    reads = seq_reads[article]
    reads_sorted = sorted(reads.items(), key=lambda kv:kv[1], reverse=True)
    if len(reads_sorted) < 3: tops = len(reads_sorted)
    else: tops = 3
    seq_read[article] = []
    for i in range(tops):
      if reads_sorted[i][1] < 2: break
      seq_read[article].append(reads_sorted[i][0])

  for article in prev_reads:
    reads = prev_reads[article]
    reads_sorted = sorted(reads.items(), key=lambda kv:kv[1], reverse=True)
    if len(reads_sorted) < 3: tops = len(reads_sorted)
    else: tops = 3
    prev_read[article] = []
    for i in range(tops):
      if reads_sorted[i][1] < 2: break
      prev_read[article].append(reads_sorted[i][0])


# may need to write pickle for this
# prepare article info
def read_article_meta():
  print("build article id and registration time for each writer")
  with open("res/metadata.json", "r") as fp:
    for line in fp:
      article = json.loads(line)
      article_id = article['id']
      writer_id = article['user_id']
      reg_datetime = datetime.datetime.fromtimestamp(article['reg_ts']/1000).strftime("%Y%m%d%H%M%S")
      if writer_id in writer_articles:
        writer_articles[writer_id].append([article_id, reg_datetime])
      else:
        writer_articles[writer_id] = [[article_id, reg_datetime]]


# dedup_recs are in reverse order (the sooner the better)
def prepare_dedup_recs():
  print("prepare recommendations with old read or no read")
  dedup_recs = []
  for writer in writer_articles:
    articles = writer_articles[writer]
    if len(articles) < 2: continue
    for item in articles:
      dedup_recs.append(item)

  dedup_recs_sorted = sorted(dedup_recs, key=lambda x: x[1], reverse=True)
  dedup_recs = []
  for article, reg_datetime in dedup_recs_sorted:
    dedup_recs.append(article)

  return dedup_recs


def add_dedup_recs(viewer, rec100, dedup_recs):
  rec100_org = rec100.copy()
  if viewer in t_reads:
    reads = t_reads[viewer]
    writers = {}
    for read in reads:
      writer = read.split("_")[0]
      if writer not in writers:
        writers[writer] = 1

    i = 0
    while i < len(dedup_recs):
      writer = dedup_recs[i].split("_")[0]
      if (dedup_recs[i] not in all_recs) and (dedup_recs[i] not in rec100) and (writer in writers):
        rec100.append(dedup_recs[i])
      i += 1
      if len(rec100) >= 100:
        break
  i = 0
  while i < len(dedup_recs):
    if len(rec100) >= 100:
      break
    if (dedup_recs[i] not in all_recs) and (dedup_recs[i] not in rec100):
      rec100.append(dedup_recs[i])
    i += 1
    if len(rec100) >= 100:
      break
  return rec100


def add_dedup_recs_d2v(viewer, rec100, dedup_recs):
  top_writers = {}
  if viewer in model.docvecs:
    tops = model.docvecs.most_similar(viewer, topn=200)
    for top in tops:
      top_writers[top[0]] = 1

  if len(top_writers) > 0:
    i = 0
    recs = []
    while i < len(dedup_recs):
      rec = dedup_recs[i]
      rec_writer = rec.split("_")[0]
      if rec_writer in top_writers:
        if (rec not in all_recs) and (rec not in rec100):
          rec100.append(rec)
      i += 1
      if len(rec100) >= 100:
        break

  if len(rec100) < 100:
    i = 0
    while i < len(dedup_recs):
      if (dedup_recs[i] not in all_recs) and (dedup_recs[i] not in rec100):
        rec100.append(dedup_recs[i])
      i += 1
      if len(rec100) >= 100:
        break

  return rec100


def most_similar(user_id, size):
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


def similar(user_id, writer_id):
    user_index = sentences_df_indexed.loc[user_id]['index']
    writer_index = sentences_df_indexed.loc[writer_id]['index']
    dist = final_doc_embeddings[user_index].dot(final_doc_embeddings[writer_index])
    #print('{} - {} : {}'.format(user_id, writer_id, dist))
    return dist

def similarity(user_id, writer_id):
    if user_id in sentences_df_indexed.index and writer_id in sentences_df_indexed.index:
        user_index = sentences_df_indexed.loc[user_id]['index']
        writer_index = sentences_df_indexed.loc[writer_id]['index']
        sim = spatial.distance.cosine(final_doc_embeddings[user_index], final_doc_embeddings[writer_index])
        #print('{} - {} : {}'.format(user_id, writer_id, sim))
        return sim

if __name__ == "__main__":
  if len(sys.argv) < 2:
    user_file = "res/predict/test.users"
  elif sys.argv[1] == "test":
    user_file = "res/predict/test.users"
  else:
    user_file = "res/predict/dev.users"

  print("load d2v model")
  #files = glob.glob('./res/writer_user_doc.txt')
  files = glob.glob('./res/writer_user_sentences_keyword.txt')
  words = []
  for f in files:
      file = open(f)
      words.append(file.read())
      file.close()

  words = list(chain.from_iterable(words))
  words = ''.join(words)[:-1]
  sentences = words.split('\n')
  sentences_df = pd.DataFrame(sentences)
  sentences_df['user'] = sentences_df[0].apply(lambda x : x.split()[0])
  sentences_df['words'] = sentences_df[0].apply(lambda x : ' '.join(x.split()[1:]))
  sentences_df_indexed = sentences_df.reset_index().set_index('user')

  #final_doc_embeddings = np.load('./doc_embeddings.npy')
  final_doc_embeddings = np.load('./doc_embeddings_keyword.npy')

  t_users = {}   # all test_users
  t_keywords = {} # keywords
  t_followings = {}   # following writer list for test users
  t_non_follows = {}   # non-follow but many reads writer list for test users
  t_non_follow = {}   # top3 non-follow but many reads writer list for test users
  t_reads = {}        # read articles for test users
  t_reads_dup = {}    # read articles during dup dates for test users (2/22~)
  writer_articles = {}
  seq_reads = {}      # sequentially read articles
  seq_read = {}       # top3 sequentially read articles
  prev_reads = {}      # sequentially read articles
  prev_read = {}       # top3 sequentially read articles
  all_recs = {}

  read_test_user()

  read_followings()

  read_reads()

  determine_seq_read()
  determine_non_follow()

  read_article_meta()
  dedup_recs = prepare_dedup_recs()

  of1 = open("res/recommend_1.txt", "w")
  of2 = open("res/recommend_2.txt", "w")
  of12 = open("recommend.txt", "w")
  print("start recommending articles")
  num_recommended = 0
  num_recommended1and2 = 0
  num_recommended1 = 0
  num_recommended2 = 0
  num_recommends1 = 0
  num_recommends2 = 0
  num_recommends1or2 = 0
  for cnt, viewer in enumerate(t_users):
    if (cnt % 100) == 99: print(str(cnt+1), "/", str(len(t_users)))
    
    recommends11, recommends12 = find_dup_seq(viewer)
    recommends1 = recommends11 + recommends12
    if len(recommends1) > 0:
      of1.write(viewer + " " + " ".join(recommends1[:100]) + "\n")
    num_recommend1 = len(recommends1[:100])
    num_recommends1 += num_recommend1

    recommends21, recommends22 = find_new_articles(viewer)
    recommends2 = recommends21 + recommends22

    if len(recommends2) > 0:
      of2.write(viewer + " " + " ".join(recommends2[:100]) + "\n")
    num_recommend2 = len(recommends2[:100])
    num_recommends2 += num_recommend2

    if num_recommend1 > 0:
      if num_recommend2 > 0:
        num_recommended1and2 += 1
      else:
        num_recommended1 += 1
    elif num_recommend2 > 0:
      num_recommended2 += 1
   
    recommends_1or2 = recommends11.copy()

    for rec in recommends21:
      if rec not in recommends_1or2:
        recommends_1or2.append(rec)

    for rec in recommends12:
      if rec not in recommends_1or2:
        recommends_1or2.append(rec)

    for rec in recommends22:
      if rec not in recommends_1or2:
        recommends_1or2.append(rec)

    num_recommends1or2 += len(recommends_1or2[:100])
    num_recommended += 1

    if len(recommends_1or2[:100]) < 100:
      #recommends_1or2 = add_dedup_recs_d2v(viewer, recommends_1or2, dedup_recs)
      recommends_1or2 = add_dedup_recs(viewer, recommends_1or2, dedup_recs)

    if len(recommends_1or2) < 100:
      pdb.set_trace()

    for rec in recommends_1or2[:100]:
      if rec not in all_recs:
        all_recs[rec] = 1

    of12.write(viewer + " " + " ".join(recommends_1or2[:100]) + "\n")

  of1.close()
  of12.close()
  of2.close()

  print(num_recommended, num_recommended1and2, num_recommended1, num_recommended2)
  print(num_recommends1, num_recommends1or2-num_recommends1)
