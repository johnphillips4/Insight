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
