# -*- coding: utf-8 -*-
"""
Created on Sat Jul 29 20:54:55 2023

@author: yonat
"""

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
from tqdm import tqdm

'''
This is a companion to 'Generative Interpretation' by Arbel & Hoffman

'''


#font_prop.set_size(15)  # Increase the font size

# Define data
animals = ['Whale', 'Tuna', 'Snake', 'Ostrich', 'Horse', 'Cow', 'Sea Turtle', 'Spider']
feet = [0, 0, 0, 2, 4, 4, 4, 8]
habitat = [0, 0, 1, 1, 1, 1, 0, 1]  # 0 = sea, 1 = land

# Adjust 'feet' values for Cow, Horse, Tuna, and Whale to make them closer
feet_adjusted = [f  if a in ['Cow', 'Horse'] else f  if a in ['Tuna', 'Whale'] else f for a, f in zip(animals, feet)]

# Generate more noise for 'feet' values
np.random.seed(0)
feet_noisy = np.array(feet_adjusted) + np.random.normal(0, 0.07, len(feet))

# Normalize 'feet' and 'habitat' values for color mapping
feet_normalized = (np.array(feet) - min(feet)) / (max(feet) - min(feet))
habitat_normalized = (np.array(habitat) - min(habitat)) / (max(habitat) - min(habitat))

# Calculate average of normalized 'feet' and 'habitat' for color mapping
average_values = (feet_normalized + habitat_normalized) / 2

# Change color palette to 'cubehelix' 
cmap = plt.get_cmap('cubehelix')
normalize = plt.Normalize(vmin=min(average_values), vmax=max(average_values))
colors = [cmap(normalize(value)) for value in average_values]

# Create scatter plot
plt.figure(figsize=(8, 5))
plt.scatter(feet_noisy, habitat, s=100, marker='o', edgecolors='black', c=colors)

# Add labels for each point
for i in tqdm(range(len(animals)), desc='Adding labels'):
    if animals[i] == 'Tuna':
        x_offset = -0.1
        y_offset = -0.1
    else:
        x_offset = 0 if animals[i] == 'Cow' else 0.1 if animals[i] not in ['Cow', 'Horse', 'Whale'] else 0.3
        y_offset = 0.1 if animals[i] == 'Whale' else 0.1 if animals[i] not in ['Cow', 'Horse'] else 0.1
    plt.text(feet_noisy[i] + x_offset, habitat[i] + y_offset, animals[i], fontsize=10, ha='right')


# Set x and y axis labels
plt.xlabel('Number of Feet')
plt.ylabel('Habitat (0 = Sea, 1 = Land)')

# Set y-axis limits
plt.ylim(-1, 2)

# Add grid
plt.grid(True)

plt.show()