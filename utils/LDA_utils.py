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
from utils import utils
import pickle

stop = ['the','a','and','an','but','to','is','of','you','are','i','for',
    'to','in','as','as','if','work','with','it','was','on',
    'good','great','(',')','!',"'s",'people','company','k','"','``',"''","n't",'#','&','%',"'",";",'/','\\']

def format_topicality(topic_list,all_topics):
    """
    Formats topic list

    Keyword arguments:
    topic_list -- the list of topics present in a chunk of text
    all_topics -- the list of all topics present in the LDA model
    """
    present_topics = [topic_list[k][0] for k in range(len(topic_list))]
    topics_well_behaved = []
    for k in range(len(all_topics)):
        if k in present_topics:
            topics_well_behaved.append(topic_list[present_topics.index(k)][1])
        else:
            topics_well_behaved.append(0)
    return topics_well_behaved




def topics_for_one_company(df,stock,lda,d):
    """
    Formats topic list

    Keyword arguments:
    df -- pandas dataframe containing reviews
    all_topics -- the list of all topics present in the LDA model
    """

    t = lda.get_topics()


    comp = df.loc[df['Ticker Symbol']==stock]

    # Now loop through the reviews, divide the reviews into topics, and weigh topics according to sentiment    
    netter = lambda x:x['pos']-x['neg']
    analyzer = vaderSentiment.SentimentIntensityAnalyzer()
    counter = 0
    comp_topics = np.zeros(len(t))
    for i in range(len(comp)):
        active_pro = comp['PROs'].iloc[i]
        active_pro_s = re.split(r'(?:\.|!|, but|, and)',active_pro.lower())  # subdivide reviews by punctuation and conjunctions
        active_pro_s = [l for l in active_pro_s if l]  # drop empty reviews
        review_doc = [d.doc2bow(active_pro_s[j].split()) for j in range(len(active_pro_s))]
        topics = lda.get_document_topics(review_doc)
        sent = [netter(analyzer.polarity_scores(utils.clean_up(active_pro_s[i]))) for i in range(len(active_pro_s))]
        sent_normed = np.zeros(len(t))
        tot_sent = 0
        for j in range(len(sent)):
            topics_well_behaved = format_topicality(topics[j],t)
            if abs(sent[j]) > 0.1:     # We only want to count reviews with non-zero sentiment
                sent_normed += np.array([topics_well_behaved[i]*sent[j] for i in range(len(t))]) # weigh by sentiment
                tot_sent += 1
        if tot_sent > 0:
            counter +=1
            comp_topics += sent_normed

    # Return the average weighted topicality 

    return comp_topics/counter  

def topics_for_all_companies(df,lda,dictionary):
    universe = np.unique(df['Ticker Symbol'].values)
    all_topics = []
    for stock in universe:
        all_topics.append(topics_for_one_company(df,stock,lda,dictionary))
    return all_topics


def LDA_from_df(df,num_topics=4,alpha='auto'):
    flat_reviews = []
    words = []
    for review in df['PROs'].values:
        cleaned_text = re.sub('([A-Za-z][A-Z][a-z])',utils.clean_multisentence_camel_case_,review).lower()
        tokens = re.sub('(\.|\,|\d|-)',' ',cleaned_text)
        tokens = remove_stopwords(tokens)
        tokens = nltk.tokenize.word_tokenize(tokens)
        tokens = [i for i in tokens if i not in stop]
        flat_reviews.append(tokens)
    d = Dictionary(flat_reviews)
    c = [d.doc2bow(text) for text in flat_reviews]
    lda = LdaModel(c, num_topics=num_topics,alpha=alpha)
    lda.save('tmp/lda')
    pickle.dump(d,'tmp/d')
    return lda,d

class disambiguated_LDA:
    '''
    LDA tool to disambiguate topics. By iteratively running self.disambiguate on ambiguous topics, the user can
    distinguish between ambiguous topics. Note that topic ambiguity can indicate a problem with the model; look
    into this more (LDA alpha and beta)
    '''
    def __init__(self,lda,dictionary):
        self.lda = lda
        self.dictionary = dictionary
        self.root = np.arange(self.lda.num_topics)
        self.branches = []
        self.label_trees = []
        self.topics = self.lda.get_topics()
        self.original_topics = self.lda.get_topics()
        self.av = self.topics.mean(axis = 0)
        self.demeaned_topics = np.array([i - self.av for i in self.topics])
        self.find_topics()
        self.label(self.root)
        
    
    def find_topics(self):
        self._tops = []
        h1 = np.argsort(self.demeaned_topics)
        for j in h1:
            self._tops.append([d1.get(i) for i in j[:8]])
        self.tops = np.array(self._tops)
    
    def label(self,branch):
        self.branch_labels = []
        for j in self.tops[branch]:
            print(j)
            label = input("Enter a label (non-unique is good!) > ")
            self.branch_labels.append(label)
        self.branches.append(branch)
        self.label_trees.append(self.branch_labels)
        
    def disambiguate(self,branch):
        '''This de-means ambiguous topics to differentiate between them'''
        self.recent = self.demeaned_topics[branch]
        self._av = self.recent.mean(axis = 0)
        self.av_demeaned = np.array([i - self._av for i in self.recent])
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


