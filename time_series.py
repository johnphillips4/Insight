import scipy
import numpy as np
import pandas as pd
import nltk
import statsmodels.api as sm
import os
import re
import gensim
import sys
import pickle
import pyLDAvis
import warnings

from nltk.tokenize import RegexpTokenizer
from gensim.models.keyedvectors import KeyedVectors
from gensim.test.utils import common_texts
from gensim.corpora.dictionary import Dictionary
from gensim.models import LdaModel
from vaderSentiment import vaderSentiment as vaderSentiment
from gensim.parsing.preprocessing import remove_stopwords
from sklearn.ensemble import GradientBoostingRegressor

from utils import sentiment_utils
from utils import LDA_utils
from utils import time_series_utils

# 

def main(df,output_filename,rng):
	universe = np.unique(df['Ticker Symbol'].values)
	all_pros,all_cons,all_rec = time_series_utils.extract_time_series_all(df,rng)
	net = pd.DataFrame((all_pros-all_cons),index = rng,columns=universe)
	rec = pd.DataFrame((all_rec),index = rng,columns=universe)
	dtree_data = DecisionTreeDataHolder(universe = universe,
		net = net,
		rec = rec,
		rng = rng,
		num_features = 6,
		time_gap = 2,
		buffer_ = 10,
		steps_train = 10,
		steps_val = 5)
	dtree_data.create_train_test()
	model_tree = GradientBoostingRegressor(n_estimators=10)
	result_tree = model_tree.fit(dtree_data.train_X,dtree_data.train_Y)
	r2 = r2_score(dtree_data.test_Y,result_tree.predict(dtree_data.test_X))
	print('r2_score = {0:.4f}'tr(r2))
	pickle.dump(result_tree,output_filename)

if __name__ == '__main__':
	'''Takes one reqd argument'''
	#print(sys.argv)
	try:
		file_loc = sys.argv[1]
		df = pd.read_pickle(file_loc)  #can be changed to read_csv as necessary
	except IndexError:
		raise ValueError('Please specify dataframe location.')
	try: 
		output_filename = sys.argv[2]
	except IndexError:
		output_filename = 'gb_model'

	rng = pd.date_range(pd.Timestamp('1-1-2014'), periods=12*4, freq='MS')
	if 'tmp' in os.listdir('.'):
		if not os.path.isdir("./tmp"):
			os.rename('./tmp','./tmp_old')
			os.mkdir('./tmp')
	else:
		os.mkdir('./tmp')

	main(df,output_filename,rng)
