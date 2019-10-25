import pandas as pd
import numpy as np
import nltk
import re          
import string
from nltk.corpus import stopwords
from scipy.spatial import distance
stop_words = stopwords.words('english')
from nltk.stem import WordNetLemmatizer 
from nltk.corpus import wordnet
import csv

def get_wordnet_pos(pos_tag):
    if pos_tag.startswith('J'):
        return wordnet.ADJ

    elif pos_tag.startswith('V'):
        return wordnet.VERB
    
    elif pos_tag.startswith('N'):
        return wordnet.NOUN
    
    elif pos_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def tokenize (text):
    token_count= None
    text= text.lower()
    pattern= r'[a-z0-9][a-z0-9-_.@]*[a-z0-9]'
    rp=nltk.regexp_tokenize(text, pattern)
    
    wordnet_lemmatizer = WordNetLemmatizer()
    
    tagged_tokens= nltk.pos_tag(rp)
    
    lemmatized_words=[wordnet_lemmatizer.lemmatize          (word, get_wordnet_pos(tag))           for (word, tag) in tagged_tokens           if word not in stop_words and           word not in string.punctuation]
    
    
    

    token_count=nltk.FreqDist(lemmatized_words)
    
    return token_count

def find_similar_doc(doc_id, docs):

    best_matching_doc_id = None
    similarity = None
    list_1 = []
    doc_dict = {}
    for key,element in enumerate(docs):
        dictornary = {}
        for key, value in tokenize(element).items():
            dictornary[key] = value
        list_1.append(dictornary)
    for i,value in enumerate(list_1):
        doc_dict[i] = value
    
    dtm=pd.DataFrame.from_dict(doc_dict, orient="index" )
    dtm=dtm.fillna(0)
             
    tf=dtm.values
    doc_len=tf.sum(axis=1)
    tf=np.divide(tf.T, doc_len).T
    
    df=np.where(tf>0,1,0)

    smoothed_idf=np.log(np.divide(len(docs)+1, np.sum(df, axis=0)+1))+1    
    smoothed_tf_idf=tf*smoothed_idf
    
    similarity=1-distance.squareform(distance.pdist(smoothed_tf_idf, 'cosine'))
    
    best_matching_doc_id = np.argsort(similarity)[:,::-1][doc_id,0:2]
    
    return best_matching_doc_id
        
    


if __name__ == "__main__":    
    

    
    text='''contact Yahoo! at "http://login.yahoo.com", 
    select forgot your password. If that fails to reset, contact Yahoo! at             
    their password department 408-349-1572 -- Can't promise             
    their phone department will fix, but they'll know where to             
    go next. Corporate emails from Yahoo! don't come from             
    their free mail system address space. Webmaster@yahoo.com             
    is not a corporate email address.'''
    
    print("Test Q1")    
    for key, value in tokenize(text).items():        
        print(key, value)
             
    data=pd.read_csv("qa.csv", header=0) 
    doc_id = 15
#     find_similar_doc(doc_id, data["question"].values.tolist())
    x,y=find_similar_doc(doc_id, data["question"].values.tolist())
    print(x,y)
    print(data["question"].iloc[doc_id])
    print(data["question"].iloc[x])

    
    doc_id=51
    x,y=find_similar_doc(doc_id, data["question"].values.tolist())
    print(x,y)
    print(data["question"].iloc[doc_id])
    print(data["question"].iloc[x])






