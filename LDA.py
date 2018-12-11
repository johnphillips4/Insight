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
from utils import sentiment_utils
from utils import LDA_utils

if __name__ == '__main__':
	'''Takes one necessary argument'''
	print(sys.argv)
	try:
		file_loc = sys.argv[1]
		df = pd.read_pickle(file_loc)  #can be changed to read_csv as necessary
	except IndexError:
		raise ValueError('Please specify dataframe location.')
	try: 
		output_filename = sys.argv[2]
	except IndexError:
		output_filename = 'test.csv'
	main(df,output_filename)