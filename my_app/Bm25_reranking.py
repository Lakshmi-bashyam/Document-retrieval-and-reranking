from gensim.summarization.bm25  import BM25
from collections import OrderedDict

import json
import string

class Bm25Model:
    def __init__(self):
        with open('./dataset/generated/tf.json', 'r+') as f:
            doc_dict = json.load(f)
        f.close()
        self.doc_text = OrderedDict()
        for docid, docdetail in doc_dict.items():
            doctext = self.preprocess(" ".join(docdetail.get("text","")))
            self.doc_text[docid] = doctext

    def get_id_from_index(self, score_list, top_1000_doc_dict):
        doc_sim = {}
        for index, key in enumerate(top_1000_doc_dict.keys()):
            doc_sim[key] = score_list[index]
        return doc_sim

    def fit_model(self, top_1000, query):
        query = self.preprocess(query)
        bm25 = BM25(top_1000.values())
        score_list = bm25.get_scores(query)
        doc_sim = self.get_id_from_index(score_list, top_1000)
        return doc_sim


    def preprocess(self, text):
        text = text.lower().translate(str.maketrans('', '', string.punctuation)).split(" ")
        return text
    
    def top_1000_doc_dict(self, ranking_list):
        key_list = ranking_list
        doc_dict_copy = self.doc_text.copy()
        for docid in self.doc_text.copy().keys():
            if docid not in key_list:
                doc_dict_copy.pop(docid)
        return doc_dict_copy

