import json
import string

from gensim.summarization.bm25  import BM25
from nltk.tokenize import word_tokenize

class BM25SentenceModel:
    def __init__(self):
        with open('./dataset/generated/tf.json', 'r+') as f:
            self.doc_dict = json.load(f)
        f.close()
        # self.sent_db = []

    def preprocess(self, text):
        text = text.translate(str.maketrans('', '', string.punctuation))
        return text
    
    def convert_to_sentence(self,docid_list):
        sent_db = []
        for docid in docid_list:
            sent_db += [self.preprocess(sent) for sent in self.doc_dict.get(docid).get("text")]
        sent_word_db = [sent.split() for sent in sent_db]
        return sent_word_db

    def fit_model(self, dataset, query):
        query = self.preprocess(query).split()
        bm25 = BM25(dataset)
        sent_sim = bm25.get_scores(query)
        sent_sim_dict = self.index_to_id(sent_sim)
        return sent_sim_dict

    def index_to_id(self, score):
        sent_2_score = {}
        for index, val in enumerate(score):
            sent_2_score[index] = val
        return sent_2_score
