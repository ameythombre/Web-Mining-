from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
from nltk.cluster import KMeansClusterer, cosine_distance
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import json
from numpy.random import shuffle
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np
from sklearn import metrics
import gensim
from gensim import corpora
from gensim.test.utils import common_corpus, common_dictionary
from gensim.models.ldamodel import LdaModel
from gensim.models.coherencemodel import CoherenceModel
import matplotlib.pyplot as plt

def getGroundTruthValues(testData):
    testFilteredData = []
    for item in testData.values:
        data = (item[0], item[1][0])
        testFilteredData.append(data)
    return testFilteredData


def cluster_kmean(train_file, test_file):
    trainData = json.load(open(train_file, 'r'))
    # testData = json.load(open(test_file, 'r'))
    testData = pd.read_json(test_file, orient= None)

    testFilteredData = getGroundTruthValues(testData)

    testDataFrame = pd.DataFrame(testFilteredData, columns=['Text', 'Labels'])
    testDataList = testDataFrame['Text'].tolist()


    fullList = trainData + testDataList

    k = 3
    tfidf_vect = TfidfVectorizer(stop_words="english")
    dtm = tfidf_vect.fit_transform(fullList)
    clusterer = KMeansClusterer(k, cosine_distance, repeats=20)
    clusters = clusterer.cluster(dtm.toarray(), assign_clusters=True)

    clustersPredicted = clusters[len(trainData): len(clusters)]
    testDataFrame['cluster'] = clustersPredicted
    dfa = pd.crosstab(index=testDataFrame.cluster, columns=testDataFrame.Labels)
    print(dfa)
    dfMax = dfa.idxmax(axis=1)

    cluster_dict = {0: dfMax[0], 1: dfMax[1], 2: dfMax[2]}

    predicted_target = [cluster_dict[i] for i in clustersPredicted]

    print(metrics.classification_report (testDataFrame["Labels"], predicted_target))

    # Calculating top words through centroid and printing out the top words for each cluster
    centroids = np.array(clusterer.means())
    sorted_centroids = centroids.argsort()[:, ::-1]
    voc_lookup = tfidf_vect.get_feature_names()

    for i in range(k):
        top_words = [voc_lookup[word_index] for word_index in sorted_centroids[i, :20]]
        print("Cluster %d:\n %s \n\n " % (i, "; ".join(top_words)))

def cluster_lda(train_file, test_file):
    trainData = json.load(open(train_file, 'r'))
    tf_vectorizer = CountVectorizer(max_df=0.90,                                     min_df=50, stop_words='english')
    testData = pd.read_json(test_file, orient=None)

    testFilteredData = getGroundTruthValues(testData)
    testDataFrame = pd.DataFrame(testFilteredData, columns=['Text', 'Labels'])
    test_data = testDataFrame['Text'].tolist()

    tf = tf_vectorizer.fit_transform(trainData)
    test_tf = tf_vectorizer.transform(test_data)

    tf_feature_names = tf_vectorizer.get_feature_names()

    no_topics = 3

    lda = LatentDirichletAllocation(n_components=no_topics,                                     max_iter=20, verbose=1,
                                    evaluate_every=1, n_jobs=1,
                                    random_state=70).fit(tf)

    topic_assign = lda.transform(test_tf)
    topicsPD = pd.DataFrame(topic_assign)
    dfMax = topicsPD.idxmax(axis=1)

    crossTab = pd.crosstab(index=dfMax, columns=testDataFrame.Labels)
    crossTab['max'] = crossTab.idxmax(axis=1)
    print(crossTab)
    cluster_dict={0:(crossTab['max'][0]),                      1:(crossTab['max'][1]),                      2:(crossTab['max'][2])}
    predicted_target=[cluster_dict[i] for i in dfMax]
    print(metrics.classification_report(testDataFrame.Labels,predicted_target))

    num_top_words = 20
    for topic_idx, topic in enumerate(lda.components_):
        print ("Topic %d:" % (topic_idx))
        # print out top 20 words per topic
        words = [(tf_feature_names[i], topic[i])                  for i in topic.argsort()[::-1][0:num_top_words]]
        print(words)
        print("\n")

if __name__ == "__main__":
    train_file = 'train_text.json'
    test_file = 'test_text.json'
    cluster_kmean(train_file, test_file)
    cluster_lda(train_file, test_file)

