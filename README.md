# Insight
This repository contains code useful for sentiment analysis and time series-based sentiment analysis.

sentiment.py takes a dataframe containing "Pro" and "Con" text reviews and converts them into sentiment scores on the range [-1,1] using VADER (c.f. http://comp.social.gatech.edu/papers/icwsm14.vader.hutto.pdf).

LDA.py takes a dataframe containing "Pro" text reviews and discovers the topics contained within them using gensim. It also adds a model file to the "tmp/" folder that allows the user to explore the discovered topics.

time_series.py takes a dataframe containing "Pro," "Con," and binary recommendation values and develops a gradient boosting regression tree model that predicts employee sentiment six months ahead of time. 

The "utils" folder contains modularized utiltity functions that are called by the .py files, and may be used by the user as required.

Note that the data used in this product were obtained from Glassdoor.com, so data columns correspond to elements of glassdoor reviews. Modification may be necessary for other applications.
