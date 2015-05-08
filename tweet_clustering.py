#!/usr/bin/env python

import sys
import csv
import numpy as np
import itertools
from collections import defaultdict
from numpy.linalg import svd
import sklearn.metrics
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans, AgglomerativeClustering
from event_text_classifier import tweet_reader, basic_unigram_tokenizer

######################################################################

def build_cooccur_matrix(reader=tweet_reader, mincount=20):
    rowdims = defaultdict(int)
    coldims = defaultdict(int)    
    counts = defaultdict(int)
    for label, tweet, extras in reader():
        # Tokenize the two fields:
        event_text_words = basic_unigram_tokenizer(extras['event_text'])
        tweet_words = basic_unigram_tokenizer(tweet)
        # Co-occurrence counts:
        for w1, w2 in itertools.product(tweet_words, event_text_words):
            counts[(w1,w2)] += 1
        # Row counts:
        for w in tweet_words:
            rowdims[w] += 1
        # Column counts:
        for w in event_text_words:
            coldims[w] += 1
    # Impose the min thresholds and establish the vocab:
    rownames = sorted([w for w, c in rowdims.items() if c >= mincount])
    colnames = sorted([w for w, c in coldims.items() if c >= mincount])
    mat = np.zeros((len(rowdims), len(coldims)))
    # Build the matrix:
    for i, w1 in enumerate(rownames):
        mat[i] = np.array([counts[(w1, w2)] for j, w2 in enumerate(colnames)])
    return (mat, rownames, colnames)

######################################################################

def pmi(mat=None, rownames=None, colnames=None, positive=True):
    """PMI on mat; positive=True does PPMI.rownames and colnames are
    not used, but rather only passed through to keep track of them."""
    # Joint probability table:
    p = mat / np.sum(mat, axis=None)
    # Pre-compute column sums:
    colprobs = np.sum(p, axis=0)
    # Vectorize this function so that it can be applied rowwise:
    np_pmi_log = np.vectorize((lambda x : _pmi_log(x, positive=positive)))
    p = np.array([np_pmi_log(row / (np.sum(row)*colprobs)) for row in p])   
    return (p, rownames, colnames)

def _pmi_log(x, positive=True):
    val = np.log(x) if x > 0.0 else 0.0
    return max([val,0.0]) if positive else val

def lsa(mat=None, rownames=None, colnames=None, k=100):
    """svd with a column-wise truncation to k dimensions"""
    rowmat, singvals, colmat = svd(mat, full_matrices=False)
    singvals = np.diag(singvals)
    trunc = np.dot(rowmat[:, 0:k], singvals[0:k, 0:k])
    return (trunc, rownames, colnames)

######################################################################

def cluster_tweets(reader=tweet_reader, matrix=None, output_filename=None, clusterer=KMeans(n_clusters=20)):    
    data = []
    mat, rownames, colnames = matrix
    for label, tweet, extras in reader():
        if 'RT' not in tweet and extras['event_text']:
            tweet_words = basic_unigram_tokenizer(tweet)
            vocab_overlap = [w for w in tweet_words if w in rownames]
            if vocab_overlap:
                tweet_vec = np.mean([mat[rownames.index(w)] for w in vocab_overlap], axis=0)
            d = extras
            d['tweet'] = tweet
            d['label'] = label
            d['vec'] = tweet_vec
            data.append(d)
    vecs = [d['vec'] for d in data]
    cluster_indices = clusterer.fit_predict(vecs)    
    
    header = ['label', 'cluster', "link", "tweet", "atbatnum", "event_text"]
    writer = csv.DictWriter(file(output_filename, 'w'), header)
    writer.writeheader()    
    for d, cind in zip(data, cluster_indices):
        del d['vec']
        d['cluster'] = cind
        writer.writerow(d)


def cluster_prediction_power(src_filename='temp_mat_clusters.csv'):
    mod = LogisticRegression()
    reader = csv.DictReader(file(src_filename))
    data = [(int(d['cluster']), d['label']) for d in reader]
    X, y = zip(*data)
    X = np.array(X).reshape(len(X), 1)
    print X.shape
    mod.fit_transform(X, y)
    predictions = mod.predict(X)    
    print sklearn.metrics.classification_report(y, predictions)
    
        

######################################################################

if __name__ == '__main__':

    import pickle

    print 'Building matrix ...'
    matrix = build_cooccur_matrix()
    print matrix[0]
    print 'Re-weighting matrix ...'
    matrix = pmi(mat=matrix[0], rownames=matrix[1], colnames=matrix[2])
    print 'Running LSA ...'
    matrix = lsa(mat=matrix[0], rownames=matrix[1], colnames=matrix[2], k=20)

    #pickle.dump(matrix, file('temp_mat.pickle', 'w'))    
    #matrix = pickle.load(file('temp_mat.pickle'))        

    print 'Clustering tweets'
    cluster_tweets(matrix=matrix, output_filename='temp_mat_clusters.csv', clusterer=KMeans(n_clusters=20))


    
    
    
    
    
            
        
