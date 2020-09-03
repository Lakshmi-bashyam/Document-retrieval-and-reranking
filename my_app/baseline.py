from bs4 import BeautifulSoup
import string
from collections import Counter
from math import log10, sqrt
import json
import re

class TF_IDF:
    def __init__(self):
        self.doc_dict = {}
        self.idf_dict = {}
        file = open('./dataset/trec_documents.xml','rt',encoding='utf8')
        xml_doc = file.read()
        soup = BeautifulSoup(xml_doc, 'lxml')
        for doc in soup.find_all('doc'):
            text = ""
            doc_no = str(doc.find('docno').string)
            for para in doc.find_all('p'):
                text += str(para.string).strip("\n")
            for headline in doc.find_all('headline'):
                text += str(headline.string)
            for txt in doc.find_all('text'):
                text += str(txt.string)
            text = [t for t in text.split("\n") if t != '']
            self.doc_dict[doc_no] = {"text":text}

        for doc_no, doc_details in self.doc_dict.copy().items():
            text = doc_details.get("text")
            text = " ".join(text).lower().translate(str.maketrans('', '', string.punctuation))
            word_list = text.split(" ")
            tf = Counter(word_list)
            tf.pop('', None)
            tf_max = tf.get(tf.most_common(1)[0][0])
            for k,v in tf.copy().items():
                tf[k] = tf[k] / tf_max
                self.idf_dict[k] = self.idf_dict.get(k,0) + 1
            doc_details["tf"] = tf
            self.doc_dict[doc_no] = doc_details
        idf_N = len(self.doc_dict)
        self.idf_dict = {k: log10(idf_N/v) for k, v in self.idf_dict.copy().items()}

    def return_doc_dict(self):
        return self.doc_dict

    def get_q_tf(self, word_list):
        tf = Counter(word_list)
        tf_max = tf.most_common(1)[0][1]
        for k,v in tf.copy().items():
            tf[k] = tf[k] / tf_max
        return tf
    
    def get_q_tfidf(self, q_tf):
        tfidf = []
        for word, tf in q_tf.items():
            tfidf.append(tf * self.idf_dict.get(word, 0))
        return tfidf

    def get_cosine(self, v1, v2, den_v2):
        num=0
        den_v1 = 0
        for val1, val2 in zip(v1, v2):
            num += val1*val2
            den_v1 += val1*val1
        den = sqrt(den_v1) * sqrt(den_v2)
        return num/den

    def get_doc_dict_q(self, q_tf, q_tfidf):
        doc_q_dict = {}
        for docid, val in self.doc_dict.items():
            tfidf = []
            for word in q_tf.keys():
                tf = val.get("tf", {}).get(word, 0) 
                idf = self.idf_dict.get(word, 0)
                tfidf.append(tf*idf)
            den = sum([val*val*self.idf_dict.get(word,0)*self.idf_dict.get(word,0) for word, val in val.get("tf").items()])
            cosine = self.get_cosine(tfidf, q_tfidf, den) if sum(tfidf) > 0 and sum(q_tfidf) > 0 else 0
            doc_q_dict[docid] = cosine
        return doc_q_dict
    
    def top_baseline(self, query, top):
        query = query.lower().translate(str.maketrans('', '', string.punctuation)).split(" ")
        q_tf = self.get_q_tf(query)
        q_tfidf = self.get_q_tfidf(q_tf)
        # get doc dictionary of query words
        doc_q_dict = self.get_doc_dict_q(q_tf, q_tfidf)
        top_n = sorted(doc_q_dict, key=doc_q_dict.get, reverse=True)[:top]
        return top_n


class Metric:
    
    def __init__(self):
        self.pattern = {}
        with open("./dataset/patterns.txt") as f:
            for line in f:
                line = line.split()
                if self.pattern.get(line[0],None):
                    self.pattern.get(line[0]).append(line[1])
                else:
                    self.pattern[line[0]] = [line[1]]

    def get_query_list(self):
        file = open('./dataset/test_questions.txt','rt',encoding='utf8')
        xml_doc = file.read()
        soup = BeautifulSoup(xml_doc, 'lxml')
        qlist = []
        index = 0
        for ques in soup.findAll('top'):
            data = ques.find('desc').text
            data = data.replace("Description:\n", "").rstrip("\n")
            index += 1
            qlist.append({"index": index,
                        "query": data})
        return qlist

    def check_relevance(self, query_detail, pred, doc_obj):
        doc = set()
        doc_dict = doc_obj.return_doc_dict()
        for docid in pred:
            text = " ".join(doc_dict.get(docid).get("text"))
            p = self.pattern.get(str(query_detail.get("index")))
            for regex in p:
                regex = re.compile(regex)
                matches = regex.findall(text)
                if len(matches) > 0:
                    doc.add(docid)
        # print(len(doc))
        return len(doc)/len(pred)

    def rank(self, query_detail, pred, sent_word_db):
        for index, docid in enumerate(pred, start=1):
            text = sent_word_db[docid]
            text = " ".join(text)
            # print(text)
            p = self.pattern.get(str(query_detail.get("index")))
            # print(p)
            for regex in p:
                regex = re.compile(regex)
                matches = regex.findall(text)
                if len(matches) > 0:
                    return index

    def MRR(self, rank_list):
        mrr = 0
        # rank_list = [r for r in rank_list if r]
        for rank in rank_list:
            if rank:
                mrr += (1/rank)
        return mrr/len(rank_list)



