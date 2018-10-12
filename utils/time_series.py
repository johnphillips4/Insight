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
	def __init__(self,num_features = 6,time_gap = 2,buffer_ = 10):
		self.num_features = num_features
		self.time_gap     = time_gap
		self.buffer       = buffer_ 