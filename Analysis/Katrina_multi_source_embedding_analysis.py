# -*- coding: utf-8 -*-
"""Multi-Source Embedding Analysis

This code supports the Katrina case analysis provided in Arbel & Hoffman, Generative Interpretation
"""



import datasets
from pathlib import Path
from InstructorEmbedding import INSTRUCTOR
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
from tqdm import tqdm
import os
import gc
import pandas as pd

rel_embeds = [
    "Flood caused by heavy rainfall",
    "Flood caused by a severe storm",
    "Flood caused by coastal surge",
    "Flood caused by a hurricane",
    "Flood caused by high tide",
    "Flood caused by monsoon rains",
    "Flood caused by a tsunami",
    "Flood caused by a broken levee",
    "Flood caused by a failed water main",
    "Flood caused by dam failure",
    "Flood caused by improper drainage",
    "Flood caused by construction near a water body",
    "Flood caused by deforestation",
    "Flood caused by infrastructure collapse",
    "Flood caused by irrigation canals overflow",
    "Flood caused by joy",
]
clause = ["Flood caused by water damage"]

# Set target directory
target_dir = '/scratch/yaarbel/'
datasets.config.DOWNLOADED_DATASETS_PATH = Path(target_dir)
os.environ['TRANSFORMERS_CACHE'] = target_dir

# Initialize dictionary for storing embeddings
embeddings_dict = {}
#all_terms = similar_terms + dissimilar_terms + covered_by_exclusion + not_covered_by_exclusion + words + clause
all_terms = rel_embeds + clause

#Note, the model is best run in a high-ram enviornment, ideally using a GPU. We used an A100 GPU and 250 GB of memory, thanks to the University of Alabama High Power Compute Center
try:
    print(model_instxl)
except NameError:
    model_instxl = None

model_name = 'INSTRUCTOR'
if not model_instxl:
    model_instxl = INSTRUCTOR('hkunlp/instructor-xl')

print(f'Generating embeddings for model: {model_name}...')
embeddings_dict[model_name] = {}

instruction = "Represent the legal sentence:"
for sentence in tqdm(all_terms):
    embeddings = model_instxl.encode([[instruction, sentence]])
    embeddings_dict[model_name][sentence] = embeddings[0]

try:
    print(model_gtr_t5_xl)
except NameError:
    model_gtr_t5_xl = None

model_name = 'GTR-T5-XL'
if not model_gtr_t5_xl:
    model_gtr_t5_xl = SentenceTransformer('sentence-transformers/gtr-t5-xl')

print(f'Generating embeddings for model: {model_name}...')
embeddings = model_gtr_t5_xl.encode(all_terms)
embeddings_dict[model_name] = {term: embedding for term, embedding in zip(all_terms, embeddings)}

del model_gtr_t5_xl
gc

try:
    model_gtr_t5_xxl
except NameError:
    model_gtr_t5_xxl = None

model_name = 'GTR-T5-XXL'
if not model_gtr_t5_xxl:
    model_gtr_t5_xxl = SentenceTransformer('sentence-transformers/gtr-t5-xxl')

print(f'Generating embeddings for model: {model_name}...')
embeddings = model_gtr_t5_xxl.encode(all_terms)
embeddings_dict[model_name] = {term: embedding for term, embedding in zip(all_terms, embeddings)}

del model_gtr_t5_xxl
gc

try:
    model_sent_t5_xl
except NameError:
    model_sent_t5_xl = None

model_name = 'SENTENCE_T5_XL'
if not model_sent_t5_xl:
    model_sent_t5_xl = SentenceTransformer('sentence-transformers/sentence-t5-xl')

print(f'Generating embeddings for model: {model_name}...')
embeddings = model_sent_t5_xl.encode(all_terms)
embeddings_dict[model_name] = {term: embedding for term, embedding in zip(all_terms, embeddings)}

try:
    model_e5_large_v2
except NameError:
    model_e5_large_v2 = None

model_name = 'e5largev2'
if not model_e5_large_v2:
    model_e5_large_v2 = SentenceTransformer('intfloat/e5-large-v2')

print(f'Generating embeddings for model: {model_name}...')
query_terms = ["query: " + term for term in all_terms]
embeddings = model_e5_large_v2.encode(query_terms, normalize_embeddings=True)
embeddings_dict[model_name] = {term: embedding for term, embedding in zip(all_terms, embeddings)}

try:
    model_e5_large
except NameError:
    model_e5_large = None

model_name = 'E5large'
if not model_e5_large:
    model_e5_large = SentenceTransformer('intfloat/e5-large')

print(f'Generating embeddings for model: {model_name}...')
query_terms = ["query: " + term for term in all_terms]
embeddings = model_e5_large.encode(query_terms, normalize_embeddings=True)
embeddings_dict[model_name] = {term: embedding for term, embedding in zip(all_terms, embeddings)}

try:
    model_instructor_large
except NameError:
    model_instructor_large = None

model_name = 'INSTRUCTOR_LARGE'
if not model_instructor_large:
    model_instructor_large = INSTRUCTOR('hkunlp/instructor-large')

print(f'Generating embeddings for model: {model_name}...')
instruction = "Represent the legal sentence:"
# Ensure that model_name is a key in embeddings_dict
if model_name not in embeddings_dict:
    embeddings_dict[model_name] = {}
for sentence in tqdm(all_terms):
    embeddings = model_instructor_large.encode([[instruction, sentence]])
    embeddings_dict[model_name][sentence] = embeddings[0]

try:
    model_e5_base_v2
except NameError:
    model_e5_base_v2 = None

model_name = 'E5_BASE_V2'
if not model_e5_base_v2:
    model_e5_base_v2 = SentenceTransformer('intfloat/e5-base-v2')

print(f'Generating embeddings for model: {model_name}...')
query_terms = ["query: " + term for term in all_terms]
embeddings = model_e5_base_v2.encode(query_terms, normalize_embeddings=True)
embeddings_dict[model_name] = {term: embedding for term, embedding in zip(all_terms, embeddings)}

try:
    model_gtr_t5_large
except NameError:
    model_gtr_t5_large = None

model_name = 'GTR_T5_LARGE'
if not model_gtr_t5_large:
    model_gtr_t5_large = SentenceTransformer('sentence-transformers/gtr-t5-large')

print(f'Generating embeddings for model: {model_name}...')
embeddings = model_gtr_t5_large.encode(all_terms)
embeddings_dict[model_name] = {term: embedding for term, embedding in zip(all_terms, embeddings)}

try:
    model_sent_t5_base
except NameError:
    model_sent_t5_base = None

model_name = 'SENTENCE_T5_BASE'
if not model_sent_t5_base:
    model_sent_t5_base = SentenceTransformer('sentence-transformers/sentence-t5-base')

print(f'Generating embeddings for model: {model_name}...')
embeddings = model_sent_t5_base.encode(all_terms)
embeddings_dict[model_name] = {term: embedding for term, embedding in zip(all_terms, embeddings)}

import numpy as np
from scipy.spatial.distance import cosine
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt

# Identify the clause key
clause_key = clause[0]
for model, terms in embeddings_dict.items():
    for term in terms.keys():
        if "loss caused directly" in term:
            clause_key = term
            break
    if clause_key:
        break

# Calculate cosine distances for each term from the clause
cosine_distances = {}
for model, terms in tqdm(embeddings_dict.items(), desc="Processing models"):
    cosine_distances[model] = {}
    for term, embeddings in terms.items():
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
        cosine_distances[model][term] = (cosine_distances[model][term] - min_distance) / (max_distance - min_distance)

# Extract all unique terms and remove the clause key
all_terms = set()
for model, terms in embeddings_dict.items():
    all_terms.update(terms.keys())
all_terms = list(all_terms)
all_terms.remove(clause_key)

# Calculate median distance for each term
median_distances = {}
for term in all_terms:
    distances = []
    for model in cosine_distances.keys():
        distances.append(cosine_distances[model][term])
    median_distances[term] = np.median(distances)

# Sort terms by median distance
sorted_terms = sorted(median_distances, key=median_distances.get)

# Create a figure
fig, axs = plt.subplots(len(sorted_terms), 1, figsize=(10, 2*len(sorted_terms)), constrained_layout=True, sharex=True)

# Create KDE plots for each term
for i, (ax, term) in enumerate(zip(axs, sorted_terms)):
    distances = []
    for model in cosine_distances.keys():
        distances.append(cosine_distances[model][term])
    sns.kdeplot(x=distances, ax=ax, fill=True, lw=0)
    ax.set_ylabel('') # Remove the default y-label
    ax.text(-0.05, 0.4, term, va='center', ha='right', rotation=0, transform=ax.transAxes) # Add the term as a y-label
    ax.label_outer() # Only show x-axis labels for the bottom subplot
    ax.set_yticks([]) # Remove y-ticks
    ax.plot(median_distances[term], 0.75, color='red', marker='o') # Add a red dot to indicate the median
    # Remove borders
    for spine in ax.spines.values():
        spine.set_visible(False)

plt.suptitle('Kernel Density Estimations of Cosine Distances for All Terms', fontsize=16)
plt.show()