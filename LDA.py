import scipy
import numpy as np
import pandas as pd
import nltk
import statsmodels.api as sm
import os
import re
import gensim
import sys
import pyLDAvis
import warnings

from vaderSentiment import vaderSentiment as vaderSentiment
from gensim.parsing.preprocessing import remove_stopwords
from nltk.tokenize import RegexpTokenizer
from gensim.models.keyedvectors import KeyedVectors
from gensim.test.utils import common_texts
from gensim.corpora.dictionary import Dictionary
from gensim.models import LdaModel

from utils import sentiment_utils
from utils import LDA_utils






def main(df,output_filename,num_topics,alpha):
	universe = np.unique(df['Ticker Symbol'].values)
	lda,d = LDA_utils.LDA_from_df(df,num_topics=4,alpha='auto')
	topics = LDA_utils.topics_for_all_companies(df,lda,d)
	output = pd.DataFrame(topics,index = universe)
	output.to_csv(output_filename)


if __name__ == '__main__':
	'''Takes one necessary argument'''
	#print(sys.argv)
	try:
		file_loc = sys.argv[1]
		df = pd.read_pickle(file_loc)  #can be changed to read_csv as necessary
	except IndexError:
		raise ValueError('Please specify dataframe location.')
	try: 
		output_filename = sys.argv[2]
	except IndexError:
		output_filename = 'test.csv'
	try: 
		num_topics = sys.argv[3]
	except IndexError:
		num_topics = 4
	try:
		alpha = sys.argv[4]
	except IndexError:
		alpha = 'auto'
	
	if 'tmp' in os.listdir('.'):
		if not os.path.isdir("./tmp"):
			os.rename('./tmp','./tmp_old')
			os.mkdir('./tmp')
	else:
		os.mkdir('./tmp')
	main(df,output_filename,num_topics,alpha)
