import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix, lil_matrix

cur_iterations = 3
iterations = 3


def findRel(rel_docs, true_docs):
    rel_doc = []
    not_rel_docs = []
    for i in range(len(rel_docs)):
        if(rel_docs[i] in true_docs):
            rel_doc.append(rel_docs[i])
        else:
            not_rel_docs.append(rel_docs[i])

    return rel_doc, not_rel_docs

def getSum(vec_docs, doc_index, shape):
    if(len(doc_index) == 0):
        return lil_matrix(shape)
    sum_arr = vec_docs[doc_index[0] - 1]
    for i in range(1, len(doc_index)):
        sum_arr+=vec_docs[doc_index[i] - 1]
    return sum_arr


def relevance_feedback(vec_docs, vec_queries, sim, n=10):
    cur_query_num = 1
    dict_query = {}
    global iterations
    global cur_iterations
    with open('data/med.rel', 'r') as f:
        rel_docs = []
        for line in f:
            arr = line.split(" ")
            if int(arr[0]) == cur_query_num:
                rel_docs.append(int(arr[2]))
            else:
                dict_query[cur_query_num] = rel_docs
                cur_query_num = int(arr[0])
                rel_docs = [] 
        dict_query[cur_query_num] = rel_docs

    # print(dict_query.keys())
    alpha = 0.8
    beta = 0.2
    for j in range(0, 1):
        for i in range(len(sim[0])):
            # print(i)
            rel_docs_cur = np.argsort(-sim[:, i])[:n]
            # print(rel_docs_cur, dict_query[i + 1])
            rel_docs, not_rel_docs = findRel(rel_docs_cur, dict_query[i + 1])
            # print(rel_docs, not_rel_docs)
            # break
            vec_queries[i] = vec_queries[i] + alpha*(getSum(vec_docs, rel_docs, vec_queries[i].shape)) - beta*(getSum(vec_docs, not_rel_docs, vec_queries[i].shape))


                
    # print(dict_query[3])
    # for query in vec_queries:
    #     print(query, query.shape)
    rf_sim = cosine_similarity(vec_docs, vec_queries) # change
    if(cur_iterations > 0):
        # print(cur_iterations)
        cur_iterations-=1
        rf_sim = relevance_feedback(vec_docs, vec_queries, rf_sim)


    cur_iterations = iterations
    return rf_sim


    """
    relevance feedback
    Parameters
        ----------
        vec_docs: sparse array,
            tfidf vectors for documents. Each row corresponds to a document.
        vec_queries: sparse array,
            tfidf vectors for queries. Each row corresponds to a document.
        sim: numpy array,
            matrix of similarities scores between documents (rows) and queries (columns)
        n: integer
            number of documents to assume relevant/non relevant

    Returns
    -------
    rf_sim : numpy array
        matrix of similarities scores between documents (rows) and updated queries (columns)
    """



def relevance_feedback_exp(vec_docs, vec_queries, sim, tfidf_model, n=10):
    cur_query_num = 1
    dict_query = {}
    global iterations
    global cur_iterations
    # print(vec_queries.dtype)
    # print(vec_docs.shape, vec_queries.shape)
    # print(vec_docs[40], vec_docs[40].shape)
    with open('data/med.rel', 'r') as f:
        rel_docs = []
        for line in f:
            arr = line.split(" ")
            if int(arr[0]) == cur_query_num:
                rel_docs.append(int(arr[2]))
            else:
                dict_query[cur_query_num] = rel_docs
                cur_query_num = int(arr[0])
                rel_docs = [] 
        dict_query[cur_query_num] = rel_docs

    # print(dict_query.keys())
    alpha = 0.8
    beta = 0.2

    for iterations in range(0, 1):
        for i in range(len(sim[0])):
            
            rel_docs_cur = np.argsort(-sim[:, i])[:n]
            rel_docs, not_rel_docs = findRel(rel_docs_cur, dict_query[i + 1])
            vec_queries[i] = vec_queries[i] + alpha*(getSum(vec_docs, rel_docs, vec_queries[i].shape)) - beta*(getSum(vec_docs, not_rel_docs, vec_queries[i].shape))
            vec_docs = vec_docs.toarray()
            vec_queries = vec_queries.toarray()
            top_vals = []
            for j in range(len(dict_query[i + 1])):
                doc_index = dict_query[i + 1][j]
                vals = np.argsort(-vec_docs[doc_index - 1, :])[:n]
                for k in range(len(vals)):
                    top_vals.append([vals[k], vec_docs[doc_index - 1][vals[k]]])
            top_vals = np.array(sorted(top_vals, key=lambda x: x[1], reverse = True), dtype = np.int32)[:, 0]
            for x in range(n):
                    vec_queries[i][top_vals[x]]+=vec_docs[doc_index - 1][top_vals[x]]
            
            vec_docs = csr_matrix(vec_docs)
            vec_queries = csr_matrix(vec_queries)
            
            # rel_docs_cur = np.argsort(-sim[:, i])[:n]
            # rel_docs, not_rel_docs = findRel(rel_docs_cur, dict_query[i + 1])
            # vec_queries[i] = vec_queries[i] + alpha*(getSum(vec_docs, rel_docs, vec_queries[i].shape)) - beta*(getSum(vec_docs, not_rel_docs, vec_queries[i].shape))
            # vec_docs = vec_docs.toarray()
            # vec_queries = vec_queries.toarray()
            # for j in range(len(dict_query[i + 1])):
            #     doc_index = dict_query[i + 1][j]
            #     top_vals = np.argsort(-vec_docs[doc_index - 1, :])[:n]
            #     for x in range(len(top_vals)):
            #         vec_queries[i][top_vals[x]]+=vec_docs[doc_index - 1][top_vals[x]]

            # vec_docs = csr_matrix(vec_docs)
            # vec_queries = csr_matrix(vec_queries)


                
    # print(dict_query[3])
    # for query in vec_queries:
    #     print(query, query.shape)
    rf_sim = cosine_similarity(vec_docs, vec_queries) # change
    if(cur_iterations > 0):
        cur_iterations-=1
        rf_sim = relevance_feedback_exp(vec_docs, vec_queries, rf_sim, tfidf_model)
    
    cur_iterations = iterations
    return rf_sim
    """
    relevance feedback with expanded queries
    Parameters
        ----------
        vec_docs: sparse array,
            tfidf vectors for documents. Each row corresponds to a document.
        vec_queries: sparse array,
            tfidf vectors for queries. Each row corresponds to a document.
        sim: numpy array,
            matrix of similarities scores between documents (rows) and queries (columns)
        tfidf_model: TfidfVectorizer,
            tf_idf pretrained model
        n: integer
            number of documents to assume relevant/non relevant

    Returns
    -------
    rf_sim : numpy array
        matrix of similarities scores between documents (rows) and updated queries (columns)
    """