# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 10:44:52 2021

@author: BoosamG
"""

#TNEB Regression
# =============================================================================
# import arff, numpy as np
# dataset = arff.load(open('explore/TNEB/eb.arff', 'rb'))
# data = np.array(dataset['data'])
# =============================================================================

from scipy.io import arff
import pandas as pd

data = arff.loadarff('explore/TNEB/eb.arff')
df = pd.DataFrame(data[0])

df.head()