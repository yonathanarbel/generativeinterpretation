# -*- coding: utf-8 -*-
"""
Created on Sat Jul 29 21:10:32 2023

@author: yonat
"""

'''

Visualization. The responses are hardcoded here but the originals are available on the repo

'''

# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt

# Reinterpret the data
def interpret_data(model_answers, model_scores):
    yes_counts = np.sum([int(score.split('/')[0]) for score, answer in zip(model_scores, model_answers) if answer == 'Yes'])
    no_counts = np.sum([int(score.split('/')[0]) for score, answer in zip(model_scores, model_answers) if answer == 'No'])
    return yes_counts, no_counts

# Data for Davinci-003, Turbo-GPT3.5, GPT-4, and Claude 2
davinci = np.array(['No']*102 + ['Yes']*98)
davinci_scores = ['9/10']*200

turbo_gpt_35 = np.array(['No']*180 + ['Yes']*20)
turbo_gpt_35_scores = ['9/10']*200

gpt_4 = np.array(['No']*200)
gpt_4_scores = ['9/10']*200

claude_2 = np.array(['No']*200)
claude_2_scores = ['9/10']*200

# Calculate counts for each model
davinci_yes_counts, davinci_no_counts = interpret_data(davinci, davinci_scores)
turbo_gpt_35_yes_counts, turbo_gpt_35_no_counts = interpret_data(turbo_gpt_35, turbo_gpt_35_scores)
gpt_4_yes_counts, gpt_4_no_counts = interpret_data(gpt_4, gpt_4_scores)
claude_2_yes_counts, claude_2_no_counts = interpret_data(claude_2, claude_2_scores)

# Calculate percentages for each model
total_davinci = davinci_yes_counts + davinci_no_counts
total_turbo_gpt_35 = turbo_gpt_35_yes_counts + turbo_gpt_35_no_counts
total_gpt_4 = gpt_4_yes_counts + gpt_4_no_counts
total_claude_2 = claude_2_yes_counts + claude_2_no_counts

bars1_percentage = [davinci_no_counts/total_davinci * 100, turbo_gpt_35_no_counts/total_turbo_gpt_35 * 100, gpt_4_no_counts/total_gpt_4 * 100, claude_2_no_counts/total_claude_2 * 100]
bars2_percentage = [davinci_yes_counts/total_davinci * 100, turbo_gpt_35_yes_counts/total_turbo_gpt_35 * 100, gpt_4_yes_counts/total_gpt_4 * 100, claude_2_yes_counts/total_claude_2 * 100]

# Define the bar width
barWidth = 0.25

# Set position of bar on X axis
r1 = np.arange(len(bars1_percentage))
r2 = [x + barWidth for x in r1]

# Reduce font size
fsize = 10
plt.rc('font', family='AGaramondPro-Regular', size=fsize)
# Import FontProperties directly

# Define colors using the cubehelix color map
bar_colors = plt.get_cmap('cubehelix')(np.linspace(0.1, 0.6, 2))
edge_colors = plt.get_cmap('cubehelix')(np.linspace(0.6, 1, 2))

# Set the colors for the bars with some transparency
bar_colors = [plt.get_cmap('cubehelix')(0.3), plt.get_cmap('cubehelix')(0.0)] # adjust colors

# Make the plot
bars1 = plt.bar(r1, bars1_percentage, color=bar_colors[0], alpha=0.7, width=barWidth, edgecolor='black', label='No')
bars2 = plt.bar(r2, bars2_percentage, color=bar_colors[1], alpha=0.7, width=barWidth, edgecolor='black', label='Yes')

# Add xticks on the middle of the group bars
plt.xlabel('Model', fontproperties=font_prop, fontweight='bold', size=15)
plt.xticks([r + barWidth for r in range(len(bars1_percentage))], ['DaVinci-003', 'Turbo-GPT3.5', 'GPT-4', 'Claude 2'], fontproperties=font_prop, fontsize=10)

# Add labels and title
plt.ylabel('Percentage (%)', fontproperties=font_prop, size=15)
plt.title('May EMI Subtract Its Own Fees?', fontproperties=font_prop, size=15)

# Create legend
plt.legend(prop=font_prop, fontsize=10)

# Increase plot dpi
plt.figure(dpi=300)

# Save and display the plot as a high-resolution PNG
plt.savefig('plot.png', dpi=300, bbox_inches='tight')
plt.show()