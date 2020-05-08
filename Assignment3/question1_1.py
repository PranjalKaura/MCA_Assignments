import nltk
import itertools
import tensorflow as tf
import numpy as np
from keras.preprocessing import sequence
nltk.download('abc')
nltk.download('stopwords')
from nltk.corpus import stopwords
corpus = nltk.corpus.abc.words()
corpus_length = 20000
stop_words = set(stopwords.words('english'))

def format_corpus(corpus, dict_length):
  dict_1 = {}
  for word in corpus:
    if word in stop_words or len(word) < 3:
      continue
    if word in dict_1.keys():
      dict_1[word]+=1
    else:
      dict_1[word] = 1
  
  dict_1 = sorted(dict_1.items(), key=lambda item: item[1], reverse = True)

  dict_2 = {}
  index = 1
  dict_3 = {}
  for i in range(dict_length - 1):
    if(dict_1[i][0] not in dict_2.keys()):
      dict_2[dict_1[i][0]] = index
      dict_3[index] = dict_1[i][0]
      index+=1

  for word in corpus:
    if word in stop_words or len(word) < 3:
      continue
    if word not in dict_2.keys():
      dict_2[word] = 0
       
  corpus_encoded = []
  for word in corpus:
    if word in stop_words or len(word) < 3:
      continue
    corpus_encoded.append(dict_2[word])

  return corpus_encoded, dict_3, dict_2


corpus_encoded, reverse_dict, corpus_dict = format_corpus(corpus, corpus_length)

window_size = 3
vector_dim = 600
epochs = 2000000

print(len(corpus_encoded), len(corpus))

sampling_table = sequence.make_sampling_table(corpus_length)
couples, labels = sequence.skipgrams(corpus_encoded, corpus_length, window_size=window_size, sampling_table=sampling_table)
word_target, word_context = zip(*couples)
word_target = np.array(word_target, dtype="int32")
word_context = np.array(word_context, dtype="int32")

# print(couples[:10], labels[:10])

from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import dot
from tensorflow.keras.layers import multiply
from tensorflow.keras.layers import subtract
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model

from sklearn.manifold import TSNE
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D

embedding = Embedding(corpus_length, vector_dim, input_length=1)
reshape = Reshape((vector_dim, 1))

input_main = Input((1,))
input_search_words = Input((1,))



main_embedding = reshape(embedding(input_main))
search_words_embedding = reshape(embedding(input_search_words))
similarity = Reshape((1,))(dot([main_embedding, search_words_embedding], axes=1, normalize = True))
output = Dense(1, activation='sigmoid')(similarity)
# output2 = dot([main_embedding, search_words_embedding], axes=0)

subtract_layer =  subtract([main_embedding, search_words_embedding]) 
eucledian_output = multiply([subtract_layer, subtract_layer])

# eucledian_output = dot([main_embedding, search_words_embedding], axes=1, normalize = True)


model = Model(inputs=[input_main, input_search_words], outputs=output)
model.compile(loss='binary_crossentropy', optimizer='adam')

Euclid_Model = Model(inputs=[input_main, input_search_words], outputs=eucledian_output)
Embeddings = Model(inputs = [input_main], outputs = main_embedding)

#The creation of the model has been refernced from the following link - https://adventuresinmachinelearning.com/word2vec-keras-tutorial/


def sum_values(pred):
  ans = 0
  for i in range(len(pred)):
    ans+=pred[i]
  return ans


def findEmbeddings(word):
  word_index = corpus_dict[word]
  in_arr_1 = np.zeros((1,))
  return Embeddings.predict_on_batch([in_arr_1])[0].flatten()

def findNeighbours(num_neighbours, word, printFlag):
  word_index = corpus_dict[word]
  sim_value_dict = {}
  in_arr_1 = np.zeros((1,))
  in_arr_2 = np.zeros((1,))
  for index in reverse_dict.keys():
    in_arr_1[0,] = word_index
    in_arr_2[0,] = index
    pred_sum = sum_values(Euclid_Model.predict_on_batch([in_arr_1, in_arr_2])[0])[0]
    # print(pred_sum)
    sim_value_dict[index] = pred_sum

  sorted_val = sorted(sim_value_dict.items(), key=lambda item: item[1])
  # print(sorted_val)
  neighbours = []
  for i in range(num_neighbours):
    neighbours.append(reverse_dict[sorted_val[i][0]])
    if(printFlag == 1):
      print(neighbours[i])

  if(printFlag == 1):
    print("----------------------------------")
  return neighbours


def tsne_plot_similar_words(title, labels, embedding_clusters, word_clusters, a):
    fig = plt.figure(figsize=(16, 9))
    colors = cm.rainbow(np.linspace(0, 1, len(labels)))
    ax = fig.add_subplot(111, projection='3d')
    for label, embeddings, words, color in zip(labels, embedding_clusters, word_clusters, colors):
        x = embeddings[:, 0]
        # print(x.shape)
        y = embeddings[:, 1]
        z = embeddings[:, 2]
        ax.scatter(x, y, z,c=color, alpha=a, label=label)
        # for i, word in enumerate(words):
        #     # print(word)
        #     ax.annotate(word, alpha=0.5, xy=(x[i], y[i]), xytext=(5, 2),
        #                  textcoords='offset points', ha='right', va='bottom', size=8)
    ax.legend(loc=4)
    plt.title(title)
    ax.grid(True)
    plt.show()


def visualise(keys):
  embedding_clusters = []
  word_clusters = []
  for word in keys:
      embeddings = []
      words = []
      for similar_word in findNeighbours(8, word, 0):
          words.append(similar_word)
          embeddings.append(findEmbeddings(similar_word))
          # print(embeddings.shape)
      embedding_clusters.append(embeddings)
      word_clusters.append(words)
  embedding_clusters = np.array(embedding_clusters)
  # print(embedding_clusters.shape)
  n, m, k = embedding_clusters.shape
  tsne_model_en_2d = TSNE(perplexity=50, n_components=3, n_iter=300, random_state=32)
  embeddings_en_2d = np.array(tsne_model_en_2d.fit_transform(embedding_clusters.reshape(n * m, k))).reshape(n, m, 3)
  tsne_plot_similar_words('Barca beat Real Madrid 5-0', keys, embeddings_en_2d, word_clusters, 0.9)


#visualisation code has been referenced from the link given in the question







arr_1 = np.zeros((1,))
arr_2 = np.zeros((1,))
arr_3 = np.zeros((1,))
keys = ['Iraq' ,'war', "wheat", "letters", "Government" ]
for cnt in range(epochs):
    idx = np.random.randint(0, len(labels)-1)
    arr_1[0,] = word_target[idx]
    arr_2[0,] = word_context[idx]
    arr_3[0,] = labels[idx]
    loss = model.train_on_batch([arr_1, arr_2], arr_3)
    if cnt % 1000 == 0:
        print("Iteration {}, loss={}".format(cnt, loss))
    if cnt%100000 == 0:
      for word in keys:
        findNeighbours(8, word, 1)
      visualise(keys)
