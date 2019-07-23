#-*- encoding: utf-8 -*-
# prepration process for doc2vec
# write article_sentences.txt with all text of articles
# write writer_user_sentences.txt with keywords of writer/user

import os, sys 
import pdb 
import json
import tqdm
from konlpy.tag import Mecab
from util import iterate_data_files
reload(sys)
sys.setdefaultencoding('utf-8')

# may need to write pickle for this
def read_reads():
  print("read reads for the test set")
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
  print("read test user set", user_file)
  with open(user_file, "r") as fp:
    for line in fp:
      viewer_id = line.strip()
      t_users[viewer_id] = 1

def normalize_mecab(line):
  tokens = mecab.morphs(line)
  sentence = " ".join(tokens)
  return sentence

if __name__ == "__main__":
  if sys.argv[1] == "test":
    user_file = "res/predict/test.users"
  elif sys.argv[1] == "dev":
    user_file = "res/predict/dev.users"
  else:
    sys.exit()

  print("read test users")
  t_users = {}
  read_test_user()

  print("read reads of test users")
  t_reads = {}
  t_reads_dup = {}
  o_reads = {}
  reads_dup = {}
  read_reads()

  mecab = Mecab()
  articles = {}
  writer_articles = {}

  print("read articles and write article sentences")
  of1 = open("article_sentences.txt", "w")
  of2 = open("titles.txt", "w")
  with open("res/metadata.json", "r") as fp:
    for line in fp:
      article = json.loads(line)
      article_id = article['id']
      writer_id = article['user_id']
      title = article['title'].strip()
      sub_title = article['sub_title'].strip()
      n_title = normalize_mecab(title)
      of2.write(n_title + "\n")
      of1.write(article_id + "\t" + n_title + "\n")
      if sub_title != "":
        n_sub_title = normalize_mecab(sub_title)
        of2.write(n_sub_title + "\n")
        of1.write(article_id + "\t" + n_sub_title + "\n")
      else:
        n_sub_title = ""
      keywords = " ".join(article['keyword_list']).strip()
      if keywords != "":
        n_keywords = normalize_mecab(keywords)
        of1.write(article_id + "\t" + n_keywords + "\n")
      else:
        n_keywords = ""
      articles[article_id] = [n_title, n_sub_title, n_keywords]
      if writer_id in writer_articles:
        writer_articles[writer_id].append(article_id)
      else:
        writer_articles[writer_id] = [article_id]
  of1.close()
  of2.close()

  num_writer = 0
  num_write_article = 0
  of3 = open("writer_user_sentences.txt", "w")
  print("write writer/test_user sentences")
  for writer in writer_articles:
    of3.write(writer)
    num_writer += 1
    for article in writer_articles[writer]:
      num_write_article += 1
      title = articles[article][0]
      sub_title = articles[article][1]
      keywords = articles[article][2]
      if keywords != "":
        #of3.write(writer + "\t" + keywords + "\n")
        of3.write(" " + keywords)
      elif sub_title != "":
        #of3.write(writer + "\t" + sub_title + "\n")
        of3.write(" " + sub_title)
      else:
        #of3.write(writer + "\t" + title + "\n")
        of3.write(" " + title)
    of3.write("\n")

  num_user = 0
  num_read_article = 0
  for user in t_users:
    reads = set(t_reads_dup[user])
    if len(reads) < 20:
      reads = set(t_reads[user])
    if len(reads) < 1: continue
    num_user += 1
    of3.write(user)
    for article in reads:
      if article in articles:
        title = articles[article][0]
        sub_title = articles[article][1]
        keywords = articles[article][2]
        if keywords != "":
          #of3.write(user + "\t" + keywords + "\n")
          of3.write(" " + keywords)
        elif sub_title != "":
          #of3.write(user + "\t" + sub_title + "\n")
          of3.write(" " + sub_title)
        else:
          #of3.write(user + "\t" + title + "\n")
          of3.write(" " + title)
        num_read_article += 1
      else:
        print("article not found in meta", article)
    of3.write("\n")

  num_user = 0
  num_read_article = 0
  for user in reads_dup:
    if num_user > 20000: break
    reads = set(reads_dup[user])
    if len(reads) < 20:
      reads = set(o_reads[user])
    if len(reads) < 1: continue
    num_user += 1
    of3.write(user)
    for article in reads:
      if article in articles:
        title = articles[article][0]
        sub_title = articles[article][1]
        keywords = articles[article][2]
        if keywords != "":
          #of3.write(user + "\t" + keywords + "\n")
          of3.write(" " + keywords)
        elif sub_title != "":
          #of3.write(user + "\t" + sub_title + "\n")
          of3.write(" " + sub_title)
        else:
          #of3.write(user + "\t" + title + "\n")
          of3.write(" " + title)
        num_read_article += 1
      else:
        print("article not found in meta", article)
    of3.write("\n")

  of3.close()
  print(str(num_writer), str(num_write_article), str(num_user), str(num_read_article))
