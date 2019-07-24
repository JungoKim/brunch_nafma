#-*- encoding: utf-8 -*-
# preprare sentence for doc2vec
# write writer_user_sentences.txt with keywords of writer/user

import os, sys 
import pdb 
import json
import tqdm
from util import iterate_data_files

# may need to write pickle for this
def read_reads():
  files = sorted([path for path, _ in iterate_data_files('2018100100', '2019030100')])
  for path in tqdm.tqdm(files, mininterval=1):
    date = path[11:19]
    for line in open(path):
      tokens = line.strip().split()
      user_id = tokens[0]
      reads = tokens[1:]
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
      else:
        if user_id in reads:
          o_reads[user_id] += reads
        else:
          o_reads[user_id] = reads
        if date >= "20190222":
          if user_id in reads_dup:
            reads_dup[user_id] += reads
          else:
            reads_dup[user_id] = reads

def read_test_user():
  with open(user_file, "r") as fp:
    for line in fp:
      viewer_id = line.strip()
      t_users[viewer_id] = 1

if __name__ == "__main__":

  user_file = "res/predict/test.users"
  print("read test users")
  t_users = {}
  read_test_user()

  print("read reads of all users")
  t_reads = {}
  t_reads_dup = {}
  o_reads = {}
  reads_dup = {}
  read_reads()

  articles = {}
  writer_articles = {}

  print("read articles metadata")
  with open("res/metadata.json", "r") as fp:
    for line in fp:
      article = json.loads(line)
      article_id = article['id']
      writer_id = article['user_id']
      title = article['title'].strip()
      sub_title = article['sub_title'].strip()
      keywords = " ".join(article['keyword_list']).strip()
      articles[article_id] = [title, sub_title, keywords]
      if writer_id in writer_articles:
        writer_articles[writer_id].append(article_id)
      else:
        writer_articles[writer_id] = [article_id]

  print("write writer sentences")
  num_writer = 0
  num_write_article = 0
  of3 = open("res/writer_user_sentences_keyword.txt", "w")
  for writer in writer_articles:
    num_writer += 1
    line = ""
    recent_articles = writer_articles[writer]
    for article in recent_articles:
      num_write_article += 1
      title = articles[article][0]
      sub_title = articles[article][1]
      keywords = articles[article][2]
      if keywords != "":
        line += " " + keywords
    if line != "": of3.write(writer + line + "\n")

  print("write user sentences")
  num_user = 0
  num_read_article = 0
  for user in t_users:
    reads = set(t_reads_dup[user])
    if len(reads) < 20:
      reads = set(t_reads[user])
    if len(reads) < 1: continue
    num_user += 1
    line = ""
    for article in reads:
      if article in articles:
        title = articles[article][0]
        sub_title = articles[article][1]
        keywords = articles[article][2]
        if keywords != "":
          line += " " + keywords
        num_read_article += 1

    if line != "": of3.write(user + line + "\n")

  for user in reads_dup:
    if num_user > 14828: break
    reads = set(reads_dup[user])
    if len(reads) < 20:
      reads = set(o_reads[user])
    if len(reads) < 1: continue
    num_user += 1
    line = ""
    for article in reads:
      if article in articles:
        title = articles[article][0]
        sub_title = articles[article][1]
        keywords = articles[article][2]
        if keywords != "":
          line += " " + keywords
        num_read_article += 1
    if line != "": of3.write(user + line + "\n")

  of3.close()
  #print(str(num_writer), str(num_write_article), str(num_user), str(num_read_article))
  print("all done!")
