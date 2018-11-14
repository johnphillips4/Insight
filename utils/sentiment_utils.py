import numpy as np
import pandas as pd
import nltk
import statsmodels.api as sm
import os
import re
import gensim
import sklearn
import pyLDAvis.gensim as gensimvis
from nltk.tokenize import RegexpTokenizer
from gensim.models.keyedvectors import KeyedVectors
from gensim.test.utils import common_texts
from gensim.corpora.dictionary import Dictionary
from gensim.models import LdaModel
from vaderSentiment import vaderSentiment as vaderSentiment




def clean_up(review):
    tokenable = re.sub('([A-Za-z][A-Z][a-z])',clean_multisentence_camel_case_,review).lower()
    return tokenable

def clean_multisentence_camel_case_(bit_of_string):
    j=1
    return bit_of_string.group(0)[:j]+' '+bit_of_string.group(0)[j:]

def sentiment_company(company,df):
    netter = lambda x:x['pos']-x['neg']
    analyzer = vaderSentiment.SentimentIntensityAnalyzer()
    comp = df.loc[df['Ticker Symbol']==company]
    comp = comp.replace(np.nan,' ')
    comp['Recommends Value'] = comp['Recommends Value'].replace(' ',np.nan)
    sent_pro = np.array([netter(analyzer.polarity_scores(clean_up(comp['PROs'].iloc[i]))) for i in range(len(comp))])
    sent_con = np.array([netter(analyzer.polarity_scores(clean_up(comp['CONs'].iloc[i]))) for i in range(len(comp))])
    comp = comp['Recommends Value'].dropna().values
    return sent_pro,sent_con,comp