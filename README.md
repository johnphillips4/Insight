# Insight
This repository contains code useful for sentiment analysis and time series-based sentiment analysis.

sentiment.py takes a dataframe containing "Pro" and "Con" text reviews and converts them into sentiment scores on the range [-1,1] using VADER (c.f. http://comp.social.gatech.edu/papers/icwsm14.vader.hutto.pdf).

The repository will be updated soon with code that assigns sentiment-weighted topic scores and time series analyses.

The "utils" folder contains modularized utiltity functions that are called by the .py files, and may be used by the user as required.

Note that the data used in this product were obtained from Glassdoor.com, so data columns correspond to elements of glassdoor reviews. Modification may be necessary for other applications.
