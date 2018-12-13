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
from utils import utils




def sentiment_company(df,company):
	"""
    Scores each of a company's "Pro" and "Con" reviews

    Keyword arguments:
    df -- pandas dataframe containing reviews
    company -- ticker symbol for company
    """
    netter = lambda x:x['pos']-x['neg']
    analyzer = vaderSentiment.SentimentIntensityAnalyzer()
    comp = df.loc[df['Ticker Symbol']==company] # formatting
    comp = comp.replace(np.nan,' ') # formatting
    comp['Recommends Value'] = comp['Recommends Value'].replace(' ',np.nan) # formatting
    sent_pro = np.array([netter(analyzer.polarity_scores(utils.clean_up(comp['PROs'].iloc[i]))) for i in range(len(comp))])
    sent_con = np.array([netter(analyzer.polarity_scores(utils.clean_up(comp['CONs'].iloc[i]))) for i in range(len(comp))])
    comp = comp['Recommends Value'].dropna().values
    return sent_pro,sent_con,comp