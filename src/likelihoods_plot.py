# plot_likelihoods.py
import pickle
import matplotlib.pyplot as plt
import numpy as np

# Load data
with open("likelihoods_summary_natural_scenes.pkl", "rb") as f:
    likelihoods = pickle.load(f)

print(likelihoods)

'''
l1_scores = np.array(likelihoods['l1'])
l2_scores = np.array(likelihoods['l2'])

# Remove -inf values for visualization
valid_mask = (l1_scores != float('-inf')) & (l2_scores != float('-inf'))
l1_scores = l1_scores[valid_mask]
l2_scores = l2_scores[valid_mask]

# Scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(l1_scores, l2_scores, alpha=0.7)
plt.plot([min(l1_scores.min(), l2_scores.min()), max(l1_scores.max(), l2_scores.max())],
         [min(l1_scores.min(), l2_scores.min()), max(l1_scores.max(), l2_scores.max())],
         'r--', label='L1 = L2')

plt.xlabel('L1 Log-Likelihood')
plt.ylabel('L2 Log-Likelihood')
plt.title('Neuron-wise Log-Likelihoods: L1 vs L2 Regularization')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("l1_vs_l2_likelihoods_natural_scenes_randomized.png")
plt.show()
'''