import re
from bs4 import BeautifulSoup
import string
import json
from math import log10
from collections import Counter
from gensim.summarization.bm25 import BM25
import pyltr
from nltk.corpus import stopwords
from nltk.stem.porter import *

class DataHelper:
    def __init__(self):
        self.stemmer = PorterStemmer() 


    def preprocess(self, text):
        query = []
        stopword_list = stopwords.words('english')
        if text:
            text = text.replace('\n','').lower().translate(str.maketrans('', '', string.punctuation)).split()
        else:
            text = ""
        for word in text:
            word = self.stemmer.stem(word)
            if word not in stopword_list:
                query.append(word)
        return query

    def doc_preprocess(self, text):
        if text:
            text = text.replace('\n','').lower().translate(str.maketrans('', '', string.punctuation))
        else:
            text = ""
        return text


class Dataloader:
    def __init__(self):
        f = open('./dataset/trec_documents.xml','rt',encoding='utf8')
        xml_doc = f.read()
        soup = BeautifulSoup(xml_doc, 'lxml')
        self.doc_dict = {}
        dataloader_helper = DataHelper()
        idf_title = {}
        idf_body = {}
        idf_title_body = {}
        for index, doc in enumerate(soup.find_all('doc')):
            doc_no = str(doc.find('docno').text)
            title = dataloader_helper.doc_preprocess(doc.find('headline').text if doc.find("headline") else "")
            title_tf = Counter(title.split())
            body = dataloader_helper.doc_preprocess(doc.find('text').text)
            body_tf = Counter(body.split())
            title_body = title + " " + body
            title_body_tf = Counter(title_body.split())
            self.doc_dict[index] = {
                "docid" : doc_no,
                "title": {
                    "text" : title,
                    "tf": title_tf,
                    "len": sum(title_tf.values())
                },
                "body": {
                    "text": body,
                    "tf": body_tf,
                    "len": sum(body_tf.values())
                }, 
                "title+body": {
                    "text": title_body,
                    "tf": title_body_tf,
                    "len": sum(title_body_tf.values())
                }
            }
            for word in title_tf.keys():
                idf_title[word] = idf_title.get(word, 0) + 1
            for word in body_tf.keys():
                idf_body[word] = idf_title.get(word, 0) + 1
            for word in title_body_tf.keys():
                idf_title_body[word] = idf_title.get(word, 0) + 1
        self.idf = {
            "title" : idf_title,
            "body" : idf_body,
            "title+body": idf_title_body
        }
        with open('./dataset/generated/idf.json') as f_idf:
            self.idf_C = json.loads(f_idf.read())
        f_idf.close()
        self.C = len(self.doc_dict)

    def get_tf_idf_features(self, query, doc_detail, type, tf, df, start):
        # tf_idf = {}
        doc_len = doc_detail.get(type).get("len")
        if doc_len:
            feature = {
                    start+1: sum(tf),
                    start+2: sum([log10(x + 1) for x in tf]),
                    start+3: sum([x/doc_detail.get(type).get("len")  for x in tf]),
                    start+4: sum([log10((x/doc_detail.get(type).get("len")) + 1)  for x in tf]),
                    start+5: sum([log10(self.C / x) if x else 0 for x in df]),
                    start+6: sum([log10(log10(self.C/ x)) if x else 0 for x in df]),
                    start+7: sum([log10((self.idf_C.get(word, 0)) + 1) for word in query]),
                    start+8: sum([log10(((i/doc_detail.get(type).get("len"))*(log10(self.C/j))) + 1) if i and j else 0 for i,j in zip(tf,df)]),
                    start+9: sum([i * log10(self.C/j) if i and j  else 0 for i,j in zip(tf,df)]),
                    start+10: sum([log10((x/doc_detail.get(type).get("len"))*(self.idf_C.get(w, 0)) + 1) if x else 0 for w,x in zip(query,tf)])
                }
        else:
            print("here")
            feature = {
                start+1: 0,
                start+2: 0,
                start+3: 0,
                start+4: 0,
                start+5: sum([log10(self.C / x) if x else 0 for x in df]),
                start+6: sum([log10(log10(self.C/ x)) if x else 0 for x in df]),
                start+7: sum([log10((self.idf_C.get(word, 0)) + 1) for word in query]),
                start+8: 0,
                start+9: 0,
                start+10: 0
            }
        return feature
    
    def get_bm25_features(self, bm25_val, feature_list):
        bm25_title = bm25_val.get("title")
        bm25_body = bm25_val.get("body")
        bm25_title_body = bm25_val.get("title+body")
        for docid in feature_list.keys():
            bm25_feature = {
                11: bm25_title[docid],
                12: log10(bm25_title[docid]),
                26: bm25_body[docid],
                27: log10(bm25_body[docid]),
                41: bm25_title_body[docid],
                42: log10(bm25_title_body[docid])
            }
            feature_list[docid].update(bm25_feature)
        return feature_list
    
    def extract_features_per_doc(self, query, doc_id):
        doc_detail = self.doc_dict.get(doc_id, None) 
        print(doc_detail.get("docid"))
        feature_set = {}
        count_title = []
        count_body = []
        count_title_body = []
        df_title = []
        df_body = []
        df_title_body = []
        datahelper = DataHelper()
        query = datahelper.preprocess(query)

        for word in query:
            count_title.append(doc_detail.get("title").get("tf").get(word, 0))
            count_body.append(doc_detail.get("body").get("tf").get(word, 0))
            count_title_body.append(doc_detail.get("title+body").get("tf").get(word, 0))
            df_title.append(self.idf.get("title").get(word, 0))
            df_body.append(self.idf.get("body").get(word, 0))
            df_title_body.append(self.idf.get("title+body").get(word, 0))
        # print(count_title)
        first_10 = self.get_tf_idf_features(query, doc_detail, "title", count_title, df_title, 0)
        feature_set.update(first_10)
        next15_25 = self.get_tf_idf_features(query, doc_detail, "body", count_body, df_body, 15)
        feature_set.update(next15_25)
        final_31_40 = self.get_tf_idf_features(query, doc_detail, "title+body", count_title_body, df_title_body, 30)
        feature_set.update(final_31_40)
        return feature_set

    def extract_feature(self, query):
        
        title_dataset = []
        body_dataset = []
        title_body_dataset = []
        doc_feature_dict = {}
        for key, doc_detail in self.doc_dict.items():
            # if doc_detail.get("docid") == " LA120889-0165 ":
        # for (key, doc_detail) in [(" LA120889-0165 ", self.doc_dict.get(" LA120889-0165 "))]:
            title_dataset.append(doc_detail.get("title").get("text").split(" "))
            body_dataset.append(doc_detail.get("body").get("text").split(" "))
            title_body_dataset.append(doc_detail.get("title+body").get("text").split(" "))
            doc_feature_dict[key] = self.extract_features_per_doc(query, key)
            # break
        bm25_title = BM25(title_dataset)
        bm25_body = BM25(body_dataset)
        bm25_title_body = BM25(title_body_dataset)
        doc_feature_dict = self.get_bm25_features({
            "title": bm25_title.get_scores(query.split()),
            "body": bm25_body.get_scores(query.split()),
            "title+body": bm25_title_body.get_scores(query.split())
        }, doc_feature_dict)



