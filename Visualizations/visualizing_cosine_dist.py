# -*- coding: utf-8 -*-
"""
Created on Sat Jul 29 20:57:26 2023

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

font_path = 'C:/Users/yonat/AppData/Local/Microsoft/Windows/Fonts/AGaramondPro-Regular.otf'
font_prop = FontProperties(fname=font_path)

with open('c:/zevel/embeddings_dict.pkl', 'rb') as f:
    embeddings_dict = pickle.load(f)
print("Embeddings loaded successfully.")

clause = ["A flood caused by water damage"]

clause_key = clause[0]
for model, terms in embeddings_dict.items():
    for term in terms.keys():
        if "loss caused directly" in term:
            clause_key = term
            break
    if clause_key:
        break

cosine_distances = {}
for model, terms in tqdm(embeddings_dict.items(), desc="Processing models"):
    cosine_distances[model] = {}
    for term, embeddings in terms.items():
        term = term.replace("Flood caused by ", "...")  # replace term early
        if term != clause_key:
            clause_embedding = embeddings_dict[model][clause_key]
            term_embedding = embeddings
            distance = cosine(clause_embedding, term_embedding)
            cosine_distances[model][term] = distance

# Normalize the distances using min-max normalization
for model in cosine_distances.keys():
    min_distance = min(cosine_distances[model].values())
    max_distance = max(cosine_distances[model].values())
    for term in cosine_distances[model].keys():
        cosine_distances[model][term] = (cosine_distances[model][term] - min_distance) / (max_distance - min_distance + 1e-10)

all_terms = set()
for model, terms in embeddings_dict.items():
    for term in terms.keys():
        term = term.replace("Flood caused by ", "...")  # replace term early
        all_terms.add(term)
all_terms = list(all_terms)
all_terms.remove(clause_key)

median_distances = {}
for term in all_terms:
    distances = []
    for model in cosine_distances.keys():
        distances.append(cosine_distances[model][term])
    median_distances[term] = np.median(distances)

sorted_terms = sorted(median_distances, key=median_distances.get)

norm = mcolors.Normalize(vmin=0, vmax=1)  # instantiate the Normalize object
cmap = plt.get_cmap('cubehelix')
fig, axs = plt.subplots(len(sorted_terms), 1, sharex=True, figsize=(10, 10))

for i, (ax, term) in enumerate(tqdm(zip(axs, sorted_terms), desc="Generating plots")):
    distances = []
    for model in cosine_distances.keys():
        distances.append(cosine_distances[model][term])
    plt.rc('font', family='Adobe Garamond Pro')
    sns.kdeplot(x=distances, ax=ax, fill=True, color=cmap(norm(median_distances[term])), lw=1, edgecolor='black')
    ax.text(-0.05, 0.4, term, va='center', ha='right', rotation=0, transform=ax.transAxes, fontsize=24, fontproperties=font_prop)
    ax.plot(median_distances[term], 0.75, color='red', marker='o', markersize=2.5)
    ax.set_ylabel('')
    ax.label_outer()
    ax.set_yticks([]) 
    ax.set_xticks([]) 
    for spine in ax.spines.values():
        spine.set_visible(False)

plt.suptitle('Cosine Distances from "Flood caused by water damage" \n to "Flood caused by"'' ', fontsize=24, fontproperties=font_prop)
plt.show()