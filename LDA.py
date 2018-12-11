import scipy
import numpy as np
import pandas as pd
import nltk
import statsmodels.api as sm
import os
import re
import gensim
import sys
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
from utils.sentiment_utils import *
from utils.LDA_utils import *





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
		sent = [netter(analyzer.polarity_scores(clean_up(active_pro_s[i]))) for i in range(len(active_pro_s))]
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

def main(df,output_filename,num_topics,alpha):
	universe = np.unique(df['Ticker Symbol'].values)
	lda,d = LDA_from_df(df,num_topics=4,alpha='auto')
	topics = topics_for_all_companies(df,lda,d)
	output = pd.DataFrame(topics,index = universe)
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