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

def clean_up(review):
	'''
	Many reviews have sentence structure where sentences are connected with no punctuation, e.g.
	"Hello worldHello world" This code will separate this into "hello world hello world"
	'''
	tokenable = re.sub('([A-Za-z][A-Z][a-z])',clean_multisentence_camel_case,review).lower()
	return tokenable


def clean_multisentence_camel_case_(bit_of_string):
	'''
	Function for use in clean up
	'''
	j=1
	return bit_of_string.group(0)[:j]+' '+bit_of_string.group(0)[j:]
