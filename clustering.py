import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns;sns.set(style='whitegrid')
from sklearn.cluster import AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabaz_score

# The objective of this module is to group the data from the pre-processed dataset
# Question 1
#    1- The choosen algorithm is from the hierarchical field for clustering, the Agglomerative Hierarchical,
#       this algorithm has an deterministic bottom-up approach based on a linkage criteria,
#       because of that approach this tecnique has a better performance on finding the structure of the data,
#       wich can be hard on this case because of the categorical and numeric values.
# 
#    2- Preprocessing for this step was performed in the pre-processing module, 
#       in this module columns with more than two categorical variables were transformed so that 
#       each column represents whether or not a hero has a feature (1 or 0). 
#       And the numerical data were normalized between 0 and 1. 
#       Thus all categories have the same weight within the Euclidean distance, avoiding bias.
#
# Question 2
#    -  To define the number of clusters two metrics were chosen to evaluate the clusters found by the algorithm. 
#       The metrics were the silhouette and ch-index score, the two ponder between the internal and external variance 
#       of the groups, and are already well established in the literature. In order to consider the results of the 
#       two metrics, Pareto-Optimal approach was used to find a solution that maximizes the two metrics. 
#       The result can be seen in the figure 'images/clustering_results.png'.

# Read pre-processsed dataset
df_clus = pd.read_csv('datasets/info_power_processed.csv')

X = df_clus.drop(columns='name')

print('Running algorithm ...')
# Executing the algorithm with the number of clusters going from 2 to 30
n_k = np.arange(2, 31)
models = [AgglomerativeClustering(k, linkage='average').fit(X)
          for k in n_k]

print('Running metrics ...')
# Evaluating the results with the metrics
sils = [silhouette_score(X=X, labels=ag.labels_) for ag in models]
chs = [calinski_harabaz_score(X, ag.labels_) for ag in models]

metrics = ['Calinski Harabaz Score - Maximization', 
            'Silhouette Score - Maximization']

print('Saving results ...')
f, axs = plt.subplots(1, 3, figsize=(20,8))

# Ploting the metrics and the Pareto-Optimal
for i, metric, results in zip(range(2), metrics, [chs, sils]):
    axs[i].plot(n_k, results)
    axs[i].set_xticks(n_k)
    axs[i].set_xlabel('k')
    axs[i].set_ylabel('Score')
    axs[i].set_title(metric)
    axs[i].scatter(np.argmax(results) + 2, np.max(results),
                s=200, facecolors='none', edgecolors='r',
                linewidth=2, label=f'best k: {np.argmax(results) + 2}')
    axs[i].legend()

markers = ['v', 's', 'x', '^', '3', 'P', 'D', 'd', 'X']
 
for xi, yi, k, m in zip(chs[:9], sils[:9], n_k[:9], markers):
    axs[2].scatter(xi, yi, label=f'k: {k}', marker=m)
    axs[2].set_xlabel('CH score')
    axs[2].set_ylabel('Silhouette score')
    axs[2].set_title('Optimo-Pareto: CH score x Silhouette score')

axs[2].scatter(chs[1], sils[1], s=200, facecolors='none',
               edgecolors='r', linewidth=2, label='Solution: k = 3')
  
axs[2].legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), 
             fancybox=True, shadow=True, ncol=5)

plt.tight_layout()
plt.savefig('images/clustering_results.png')
print('Done. Results in images/clustering.png')