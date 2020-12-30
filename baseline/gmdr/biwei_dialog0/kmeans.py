import time            
import re            
import os    
import sys  
import codecs  
import shutil  
from sklearn import feature_extraction    
from sklearn.feature_extraction.text import TfidfTransformer    
from sklearn.feature_extraction.text import CountVectorizer  
from sklearn.cluster import KMeans  

corpus = [] 
raw_data = []
num_none = 0
with codecs.open(sys.argv[1],'r',encoding="utf8") as f:
    for idx,line in enumerate(f):

        data = line.strip().split("\t")
        raw_data.append(data)
        text = data[0].replace(" </s>","")
        corpus.append(text)
        
def get_batch_corpus(batch_size):
    minibatch, size_so_far = [], 0
    for idx,(data,seq) in enumerate(zip(raw_data,corpus)):
        minibatch.append((data,seq))
        size_so_far = len(minibatch)
        if size_so_far == batch_size:
            yield minibatch
            minibatch, size_so_far = [], 0

def kmeans_process(batch):
    global num_none
    batch_corpus=[]
    batch_raw=[]
    for e in batch:
        batch_corpus.append(e[1])
        batch_raw.append(e[0])
    vectorizer = CountVectorizer()  
    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(vectorizer.fit_transform(batch_corpus))  
    word = vectorizer.get_feature_names()  
    weight = tfidf.toarray() 
    clf = KMeans(n_clusters=3,max_iter=300)
    s = clf.fit(weight) 
    clusters = [[] for _ in range(3)]

    for i,data in enumerate(batch_raw):
        clusters[clf.labels_[i]].append(data)
        
    clusters[0] = sorted(clusters[0],key=lambda x:-float(x[3]))
    clusters[1] = sorted(clusters[1],key=lambda x:-float(x[3]))
    clusters[2] = sorted(clusters[2],key=lambda x:-float(x[3]))
    if len(clusters[0])>0:
        cluster0 = "\t".join(clusters[0][0])
    else:
        cluster0="None"
        num_none+=1
    if len(clusters[1])>0:
        cluster1 = "\t".join(clusters[1][0])
    else:
        cluster1="None"
        num_none+=1
    if len(clusters[2])>0:
        cluster2 = "\t".join(clusters[2][0])
    else:
        cluster2="None"
        num_none+=1

    return cluster0,cluster1,cluster2
for batch in get_batch_corpus(1000):
    try:
        cluster0,cluster1,cluster2 = kmeans_process(batch)
        print(cluster0)
        print(cluster1)
        print(cluster2)
    except ValueError:
        print("ValueError")
        print("ValueError")
        print("ValueError")

print("num_none: %d"%(num_none))
