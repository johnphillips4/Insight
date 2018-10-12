import scipy
import numpy as np
import pandas as pd
import nltk
import statsmodels.api as sm
import os
import re
import gensim
from nltk.tokenize import RegexpTokenizer
from gensim.models.keyedvectors import KeyedVectors
from gensim.test.utils import common_texts
from gensim.corpora.dictionary import Dictionary
from gensim.models import LdaModel
import pyLDAvis.gensim as gensimvis
from vaderSentiment import vaderSentiment as vaderSentiment
from gensim.parsing.preprocessing import remove_stopwords
import pyLDAvis
import warnings

def clean_up(review):
	'''
	Many reviews have sentence structure where sentences are connected with no punctuation, e.g.
	"Hello worldHello world" This code will separate this into "hello world hello world"
	'''
    tokenable = re.sub('([A-Za-z][A-Z][a-z])',clean_multisentence_camel_case,review).lower()
    return tokenable


def clean_multisentence_camel_case_(bit_of_string):
	'''
	Function for use in clean up
	'''
    j=1
    return bit_of_string.group(0)[:j]+' '+bit_of_string.group(0)[j:]

class disambiguated_LDA:
    '''
    LDA tool to disambiguate topics. By iteratively running self.disambiguate on ambiguous topics, the user can
    distinguish between ambiguous topics. Note that topic ambiguity can indicate a problem with the model; look
    into this more (LDA alpha and beta)
    '''
    def __init__(self,lda,d):
        self.lda                    = lda
        self.d                      = d
        self.root                   = np.arange(self.lda.num_topics)
        self.branches               = []
        self.label_trees            = []
        self.topics                 = self.lda.get_topics()
        self.original_topics        = self.lda.get_topics()
        self.av                     = self.topics.mean(axis = 0)
        self.demeaned_topics        = np.array([i - self.av for i in self.topics])
        self.find_topics()
        self.label(self.root)
        
    
    def find_topics(self):
        tops_                       = []
        h1                          = np.argsort(self.demeaned_topics)
        for j in h1:
            tops_.append([d1.get(i) for i in j[:8]])
        self.tops                   = np.array(tops_)
    
    def label(self,branch):
        self.branch_labels = []
        for j in self.tops[branch]:
            print(j)
            label = input("Enter a label (non-unique is good!) > ")
            self.branch_labels.append(label)
        self.branches.append(branch)
        self.label_trees.append(self.branch_labels)
        
    def disambiguate(self,branch):
        self.recent                 = self.demeaned_topics[branch]
        self.av_                    = self.recent.mean(axis = 0)
        self.av_demeaned            = np.array([i - self.av_ for i in self.recent])
        for i in range(len(branch)):
            self.demeaned_topics[branch[i]] = self.av_demeaned[i]
        self.find_topics()
        self.label(branch)

    def finalize(self):
        self.final_labels = []
        for t in range(len(self.tops)):
            top = ''
            for b in range(len(self.branches)):
                if t in self.branches[b]:
                    top = top+self.label_trees[b][list(self.branches[b]).index(t)]+':'
            top = top[:-1]
            self.final_labels.append(top)


class corpus_aggregator(disambiguated_LDA):
    '''
    Class that can, given a BOW corpus and disambiguated LDA model (using on the same dictionary)
    calculate the topic distribution within the corpus, marginalizing over documents and accounting
    for documents with missing topics (which is many of them)
    '''
    def analyze_corpus(self,corpus):
        self.corpus                                = [i for i in corpus if len(i)>0]
        self.running                               = np.zeros(len(self.topics))
        self._unsanitized                          = self.lda.get_document_topics(self.corpus)
        i                                          = 0 
        for j in self._unsanitized:
            i                                      +=1
            present                                = list(np.array(j).T[0])
            for k in range(len(self.running)):
                if k in present:
                    self.running[k]                += np.array(j).T[1][present.index(k)]
        self.running = np.array(self.running)/i