# -*- coding: utf-8 -*-
"""
Created on Sun May 15 18:40:02 2022

@author: CEMRE
"""

import re
import nltk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

nltk.download("stopwords")
#%%
PLOT_PALETTE = 'tableau-colorblind10'
# for other color map, please run: mpl.pyplot.colormaps()
WORDCLOUD_COLOR_MAP = 'tab10_r'
#%%
plt.style.use(PLOT_PALETTE)
%matplotlib inline
#%%
df = pd.read_csv('C:/Users/CEMRE/Desktop/Resume.csv')
df.head()
#%%

#df.pop('ID')
#df.pop('Resume_html')
#df
#%%
pd.concat('Category')
pd.concat('Resume_str')
pd()
