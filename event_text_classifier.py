#!/usr/bin/env python

import re
import sys
import csv
import copy
import numpy as np
import itertools
from operator import itemgetter
from collections import Counter, defaultdict

from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_selection import SelectFpr, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import cross_val_score, train_test_split
from sklearn import metrics

######################################################################
# Data reader

LABEL_FIELDNAME = "event_class"
TEXT_FIELDNAME = "event_text"

def mlb_reader(src_filename="data/2014postseason.atbats.csv"):
    for d in csv.DictReader(file(src_filename)):
        yield (d[LABEL_FIELDNAME].lower().strip(), d[TEXT_FIELDNAME])

def get_sensible_classes(reader=mlb_reader, count_threshold=50):
    counts = defaultdict(int)
    for label, _ in reader():
        counts[label] += 1
    return set([label for label, count in counts.items() if count >= count_threshold])

######################################################################
# Tokenization for features

WORD_RE_STR = r"""
(?:[a-z][a-z'\-_]+[a-z])       # Words with apostrophes or dashes.
|
(?:[+\-]?\d+[,/.:-]\d+[+\-]?)  # Numbers, including fractions, decimals.
|
(?:[\w_]+)                     # Words without apostrophes or dashes.
|
(?:\.(?:\s*\.){1,})            # Ellipsis dots.
|
(?:\*{1,})                     # Asterisk runs.
|
(?:\S)                         # Everything else that isn't whitespace.
"""

WORD_RE = re.compile(r"(%s)" % WORD_RE_STR, re.VERBOSE | re.I | re.UNICODE)
 
def basic_unigram_tokenizer(s, lower=True):
    words = WORD_RE.findall(s)
    if lower:
        words = [w.lower() for w in words]
    return words

######################################################################
# Feature function

def basic_unigram_feature_function(s, lower=True):
    return Counter(basic_unigram_tokenizer(s, lower=lower))

######################################################################
# Model fitting and assessment

def featurizer(reader=mlb_reader, feature_function=basic_unigram_feature_function, count_threshold=50):
    """Map the data in reader to a list of features according to feature_function,
    and create the gold label vector. count_threshold restricts to classes with at
    least that many examples in the data."""
    classes = get_sensible_classes(reader=mlb_reader, count_threshold=count_threshold)
    feats = []
    labels = []
    split_index = None
    for label, text in reader():
        if label in classes:
            feats.append(feature_function(text))
            labels.append(label)              
    return (feats, labels)
    
def train_and_evaluate_classifier(
        reader=mlb_reader,
        count_threshold=50, # Only classes with at least this many examples.
        feature_function=basic_unigram_feature_function, # Use None to stop feature selection
        cv=5, # Number of folds used in cross-validation
        priorlims=np.arange(.1, 3.1, .1), # regularization priors to explore (we expect something around 1)
        multi_class=['ovr', 'multinomial'], # One-vs-rest or softmax loss.
        train_size=0.8): 
            
    # Featurize the data:
    feats, labels = featurizer(reader=reader, feature_function=feature_function, count_threshold=count_threshold) 
    
    # Map the count dictionaries to a sparse feature matrix:
    vectorizer = DictVectorizer(sparse=False)
    feat_matrix = vectorizer.fit_transform(feats)
    
    ##### HYPER-PARAMETER SEARCH
    # Define the basic model to use for parameter search:
    searchmod = LogisticRegression(fit_intercept=True, intercept_scaling=1, solver='lbfgs')
    # Parameters to grid-search over:
    parameters = {'C':priorlims, 'penalty':['l1','l2'], 'multi_class': multi_class}  
    # Cross-validation grid search to find the best hyper-parameters:     
    clf = GridSearchCV(searchmod, parameters, cv=cv)
    clf.fit(feat_matrix, labels)
    params = clf.best_params_

    # Establish the model we want using the parameters obtained from the search:
    mod = LogisticRegression(fit_intercept=True, intercept_scaling=1, C=params['C'], penalty=params['penalty'])

    ##### ASSESSMENT OF THE BEST MODEL WE FOUND            
    # Cross-validation of our favored model; for other summaries, use different
    # values for scoring: http://scikit-learn.org/dev/modules/model_evaluation.html
    scores = cross_val_score(mod, feat_matrix, labels, cv=cv, scoring="f1_macro")       
    print '\nBest model found in cross-validation', mod
    print 'F1 mean: %0.2f (+/- %0.2f)' % (scores.mean(), scores.std()*2)

    ##### RANDOM TRAIN/TEST SPLIT FOR EVALUATION
    train_matrix, test_matrix, train_labels, test_labels = train_test_split(feat_matrix, labels, train_size=train_size)    
    mod.fit(train_matrix, train_labels)
    test_predictions = mod.predict(test_matrix)
    #### A standard effectiveness summary, by-category
    print "\nBy-category evaluation via random train/test split"
    print metrics.classification_report(test_labels, test_predictions)


######################################################################


if __name__ == '__main__':

    # Smaller grid search to save time; the model is perfect anyway:
    #train_and_evaluate_classifier(cv=2, priorlims=[1.0], multi_class=['ovr'], train_size=0.55)
    train_and_evaluate_classifier(train_size=0.65)
    
