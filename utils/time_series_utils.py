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
import pyLDAvis.gensim as gensimvis

class decision_tree_data_holder:
	'''
	Class that holds the data from which we will make time series predictions.
	Takes

	num_features: number of previous month data to use as features
	time_gap: number of months in the future we want to predict
	buffer: amount of "burn-in" to allow
	''' 
	def __init__(self,universe,net,num_features = 6,time_gap = 2,buffer_ = 10):
		self.num_features = num_features
		self.time_gap = time_gap
		self.buffer = buffer_ 
		self.train_X,self.train_Y = [],[]
		self.test_X,self.test_Y = [],[]
		self.universe = universe
		self.net = net

		def create_train_test():
			for u in range(len(self.universe)):
			    comp = self.net[self.universe[u]]
			    for i in range(len(rng)):
			        if i > buffer_ and i < buffer_+10:
			            comp = self.net[self.universe[u]]
			            d = VZ.values[i:i+self.num_features+self.time_gap]
			            d = (d-d[self.num_features])/d[self.num_features]
			            if not np.isnan(d).any():
			                self.train_X.append(d[:self.num_features])
			                if d[-1] < -0.0035:
			                    self.train_Y.append(0.)
			                elif d[-1]> 0.0045:
			                    self.train_Y.append(2.)
			                else:
			                    self.train_Y.append(1.)
			        elif i > buffer_+10 and i < len(comp)-self.num_features-self.time_gap:
			            d = comp.values[i:i+self.num_features+self.time_gap]
			            d = (d-d[self.num_features])/d[self.num_features]
			            if not np.isnan(d).any():
			                self.test_X.append(d[:self.num_features])
			                if d[-1] < -0.0035:
			                    self.test_Y.append(0.)
			                elif d[-1]> 0.0045:
			                    self.test_Y.append(2.)
			                else:
			                    self.test_Y.append(1.)


def extract_time_series(df,rng):
	universe = np.unique(df['Ticker Symbol'].values)
	all_pros = []
	all_cons = []
	all_rec  = []
	for comp in universe[:]:
	    pros = []
	    cons = []
	    rec  = []
	    for d in rng[:]:
	        try:
	            active_pros = []
	            active_cons = []
	            active_rec  = []
	            valids = (d-df['As of Date']>pd.Timedelta('30 days 00:00:00')).values
	            doc = df.iloc[np.logical_and(df['Ticker Symbol'] == comp,valids).values]
	            doc = pd.DataFrame({'PROs':doc['PROs'],'CONs':doc['CONs'],'Rec':doc['Recommends Value']})
	            doc = doc.dropna()
	            for j in range(len(doc)):
	                cleaned_pro = re.sub('([A-Za-z][A-Z][a-z])',clean_multisentence_camel_case_,doc['PROs'].iloc[j]).lower()
	                cp = analyzer.polarity_scores(cleaned_pro)
	                p = cp['pos']-cp['neg']
	                active_pros.append(p)
	                cleaned_con = re.sub('([A-Za-z][A-Z][a-z])',clean_multisentence_camel_case_,doc['CONs'].iloc[j]).lower()
	                cc = analyzer.polarity_scores(cleaned_con)
	                c = cc['pos']-cc['neg']
	                active_cons.append(c)
	                active_rec.append(doc['Rec'].mean())
	        except TypeError:
	            active_pros.append([])
	            active_cons.append([])
	            active_rec.append([])
	        pros.append(active_pros)
	        cons.append(active_cons)
	        rec.append(active_rec)
	    all_pros.append(pros)
	    all_cons.append(cons)
	    all_rec.append(rec)
	return all_pros,all_cons,all_rec
