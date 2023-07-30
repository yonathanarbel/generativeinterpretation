# -*- coding: utf-8 -*-
"""
Created on Sat Jul 29 20:59:09 2023

@author: yonat
"""

import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors
from matplotlib.font_manager import FontProperties
from scipy.spatial.distance import cosine
from tqdm import tqdm

from matplotlib import font_manager

# Add the font directory to the font manager
font_manager.fontManager.addfont('C:/Users/yonat/AppData/Local/Microsoft/Windows/Fonts/AGaramondPro-Regular.otf')

# Now you can use the font in your plots by name
plt.rc('font', family='AGaramondPro-Regular')


fpath = "Trident_Outputs.xlsx"

# Import necessary libraries
import pandas as pd

# Load the data (modify this to match your local data path)
all_outputs = pd.read_excel(fpath)

# Define line styles for each model
line_styles = ['solid', 'dashed', 'dashdot', 'dotted']

# Get the colors from the cubehelix colormap

colors = plt.get_cmap('cubehelix')(np.linspace(0, 0.6, len(all_outputs.columns)))


fsize = 15
# Create KDE plots for each model with different line styles, colors and no x-label
plt.figure(figsize=(10, 6))

for column, style, color in zip(all_outputs.columns, line_styles, colors):
    plt.rc('font', family='Adobe Garamond Pro', size=fsize)
    sns.kdeplot(data=all_outputs, x=column, fill=True, label=column, clip=(0, 100), linestyle=style, color=color)

plt.title('On a Scale of 0-100, How Likely is Early Repayment')
plt.xlabel('')
plt.legend()
plt.show()