import argparse

import pandas as pd
from matplotlib import pyplot as plt

from sklearn.decomposition import PCA


parser = argparse.ArgumentParser()
parser.add_argument('directory_path')
parser.add_argument('--save-path', action='store')
args = parser.parse_args()

data = pd.read_csv(args.directory_path + 'tensors.tsv', sep='\t')
labels = pd.read_csv(args.directory_path + 'metadata.tsv', sep='\t')
labels = labels.values.flatten()

pca = PCA(n_components=2)
data_pca = pca.fit_transform(data)

plt.scatter(*data_pca[labels == 1].T, alpha=0.8, linewidths=0,
            label='P (pair numbers)')
plt.scatter(*data_pca[labels == 0].T, alpha=0.8, linewidths=0,
            label='N that never get labeled (7, 9)')
plt.scatter(*data_pca[labels == -1].T, alpha=0.8, linewidths=0,
            label='bN (1, 3, 5)')
plt.tight_layout()
left, right = plt.xlim()
ax = plt.gca()
ax.set_xlim(left-15, right)
plt.legend(loc=(-0.05, 0.01))
plt.axis('off')
if args.save_path is not None:
    plt.savefig(args.save_path)
plt.show()
