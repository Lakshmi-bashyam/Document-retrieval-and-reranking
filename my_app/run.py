from baseline import TF_IDF, Metric
from Bm25_reranking import Bm25Model
from sentence_ranking import BM25SentenceModel

import json

if __name__=="__main__":

# TF-IDF calculation
    metric = Metric()
    doc_obj = TF_IDF()
    bm25_model = Bm25Model()
    bm25_sent = BM25SentenceModel()
    ranking = {}
    query = metric.get_query_list()
    # rel_doc = get_relevant_doc(pattern, doc_dict) 
    mean_precision_tf = []
    mean_precision_bm25 = []
    rank = []
    for q_details in query:
        pred_tf = doc_obj.top_baseline(q_details.get("query",""), 1000)
        # ranking[q_details.get("index")] = pred
        precision_tf = metric.check_relevance(q_details, pred_tf, doc_obj)
        # print(q_details.get("query"), precision_tf)

        # BM25
        top_1000 = bm25_model.top_1000_doc_dict(pred_tf)
        doc_sim = bm25_model.fit_model(top_1000, q_details.get("query",""))
        pred_bm25 = sorted(doc_sim, key=doc_sim.get, reverse=True)[:50]
        precision_bm25 = metric.check_relevance(q_details, pred_bm25, doc_obj)
        mean_precision_bm25.append(precision_bm25)
        mean_precision_tf.append(precision_tf)

        # Sentence ranking
        sent_word_db = bm25_sent.convert_to_sentence(pred_bm25)
        sent_sim = bm25_sent.fit_model(sent_word_db, q_details.get("query",""))
        pred_bm25_sent = sorted(sent_sim, key=sent_sim.get, reverse=True)[:50]
        ranking = metric.rank(q_details, pred_bm25_sent, sent_word_db)
        print(q_details.get("query"), precision_tf, precision_bm25, ranking)
        rank.append(ranking)
        # break
    mrr = metric.MRR(rank)
    print("Mean precision of TF-IDF is ", str(sum(mean_precision_tf)/len(mean_precision_tf)))
    print("Mean precision of BM25 is ", str(sum(mean_precision_bm25)/len(mean_precision_bm25)))
    print("MRR of BM25 on sentence is", str(mrr))
    # with open('../ranking.json', 'w+') as f:
    #     json.dump(ranking, f)

