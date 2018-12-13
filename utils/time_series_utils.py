import numpy as np
import scipy
import pandas as pd
import nltk
import statsmodels.api as sm
import os
import re
import gensim
import sklearn

from nltk.tokenize import RegexpTokenizer
from gensim.models.keyedvectors import KeyedVectors
from gensim.test.utils import common_texts
from gensim.corpora.dictionary import Dictionary
from gensim.models import LdaModel
from vaderSentiment import vaderSentiment as vaderSentiment

from utils import utils

class DecisionTreeDataHolder:
    '''
    Class that holds the data from which we will make time series predictions.
    Takes
	
	Keyword Arguments - 
    universe - list of companies in our universe
    net - df containing net sentiment scores for each company for each date
    rec - df containing recommendation scores for each company for each date
    rng - pd.daterange object
    num_features - number of preceding months the model uses as features
    time_gap - time gap between the features and the prediction 
    buffer_ - amount of "burn in" allowed at the beginning of the data
	steps_train - number of dates that go into the training set
	steps_val - number of dates that go into the validation set
    ''' 
    def __init__(self,universe,net,rec,rng,num_features = 6,time_gap = 6,buffer_ = 10,steps_train = 10,steps_val = 5):
        self.num_features = num_features
        self.time_gap = time_gap
        self.buffer_ = buffer_ 
        self.train_X,self.train_Y = [],[]
        self.val_X,self.val_Y = [],[]
        self.test_X,self.test_Y = [],[]
        self.universe = universe
        self.net = net
        self.rec = rec
        self.steps_train = steps_train
        self.steps_val = steps_val
        self.rng = rng

    def create_train_test(self):
        for u in range(len(self.universe)):
            VZ = self.net[self.universe[u]]
            for i in range(len(self.rng)):
                VZ = self.net[self.universe[u]]
                VZ_rec = self.rec[self.universe[u]]
                d = VZ.values[i:i+self.num_features+self.time_gap+1]
                d_rec = VZ_rec.values[i:i+self.num_features+self.time_gap+1]
                if not np.isnan(d).any() and not np.isnan(d_rec).any():
                    if i > self.buffer_ and i < self.buffer_+self.steps_train:	
                        self.train_X.append(list(d_rec[:self.num_features])+list(d[:self.num_features]))
                        self.train_Y.append(d[-1])
                    elif i > self.buffer_+self.steps_train and i < self.buffer_ + self.steps_train+self.steps_val and i < len(VZ)-self.num_features-self.time_gap:
                        self.val_X.append(list(d_rec[:self.num_features])+list(d[:self.num_features]))
                        self.val_Y.append(d[-1])
                    elif i >=self.buffer_+self.steps_train+self.steps_val and i < len(VZ)-self.num_features-self.time_gap:
                        self.test_X.append(list(d_rec[:self.num_features])+list(d[:self.num_features]))
                        self.test_Y.append(d[-1])


def process_doc(doc):
    """
	Extract mean pros, cons, and recommendation scores from a dataframe

    Keyword arguments:
    doc -- pandas dataframe containing reviews
    """
	if doc.empty:
		return np.nan,np.nan,np.nan
	analyzer = vaderSentiment.SentimentIntensityAnalyzer()
	active_pros = []
	active_cons = []
	for j in range(len(doc)):
		cleaned_pro = re.sub('([A-Za-z][A-Z][a-z])',utils.clean_multisentence_camel_case_,doc['PROs'].iloc[j]).lower()
		cp = analyzer.polarity_scores(cleaned_pro)
		p = cp['pos']-cp['neg']
		active_pros.append(p)
		cleaned_con = re.sub('([A-Za-z][A-Z][a-z])',utils.clean_multisentence_camel_case_,doc['CONs'].iloc[j]).lower()
		cc = analyzer.polarity_scores(cleaned_con)
		c = cc['pos']-cc['neg']
		active_cons.append(c)
	pros = np.mean(active_pros)
	cons = np.mean(active_cons)
	rec = doc['Rec'].mean()
	return pros,cons,rec


def extract_time_series_one_company(df,comp,rng):
    """
	Extract time series for pros, cons, and rec for a single company over the date range rng

    Keyword arguments:
    df - dataframe containing reviews
    comp - ticker symbol for company
    rng - pd.daterange object
    """
	pros = []
	cons = []
	rec  = []
	for date in rng[:]:
		valids = (date-df['As of Date']>pd.Timedelta('30 days 00:00:00')).values
		doc = df.iloc[np.logical_and(df['Ticker Symbol'] == comp,valids).values]
		doc = pd.DataFrame({'PROs':doc['PROs'],'CONs':doc['CONs'],'Rec':doc['Recommends Value']})
		doc = doc.dropna()
		active_pros,active_cons,active_rec = process_doc(doc)
		pros.append(active_pros)
		cons.append(active_cons)
		rec.append(active_rec)
	return pros,cons,rec


def extract_time_series_all(df,rng):
    """
	Extract time series for pros, cons, and rec for a all companies over the date range rng

    Keyword arguments:
    df - dataframe containing reviews
    rng - pd.daterange object
    """

	universe = np.unique(df['Ticker Symbol'].values)
	all_pros = []
	all_cons = []
	all_rec  = []
	for comp in universe:
		pros,cons,rec = extract_time_series_one_company(df,comp,rng)
		all_pros.append(pros)
		all_cons.append(cons)
		all_rec.append(rec)
	return np.array(all_pros).T,np.array(all_cons).T,np.array(all_rec).T
