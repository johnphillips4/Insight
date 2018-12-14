import numpy as np
import pandas as pd
import nltk
import statsmodels.api as sm
import os
import re
import gensim
import sklearn
import pyLDAvis.gensim as gensimvis
import sys

from nltk.tokenize import RegexpTokenizer
from gensim.models.keyedvectors import KeyedVectors
from gensim.test.utils import common_texts
from gensim.corpora.dictionary import Dictionary
from gensim.models import LdaModel

from utils import sentiment_utils

def main(df,output_filename):
	pros = []
	cons = []
	recommendation = []
	universe = np.unique(df['Ticker Symbol'].values)
	for stock in universe:
		sents = sentiment_utils.sentiment_company(df,stock)
		pros.append(np.mean(sents[0]))
		cons.append(np.mean(sents[1]))
		recommendation.append(np.mean(sents[2]))
	output = pd.DataFrame({'Pros':pros,'Cons':cons,'Rec':recommendation},index = universe)
	output.to_csv(output_filename)

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
