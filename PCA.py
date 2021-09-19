# PCA  to reconstruct new features  #

from sklearn.decomposition import PCA
import numpy as np

pca = PCA(n_components=35).fit(features)

cum_var = np.cumsum(np.asarray(pca.explained_variance_ratio_))
for i in range(len(cum_var)):
    print([i+1], cum_var[i])

import matplotlib.pyplot as plt
pca_fig = plt.figure(figsize=(6, 6))
plt.plot(np.arange(1, pca.n_components_+1), cum_var, 'ro')
plt.show()

""" Projection """
comp = pca.components_ #35X155
com_tr = np.transpose(pca.components_) #155X35
proj = np.dot(features,com_tr) #3088X155 * 155X35 = 3088X35
""" Reconstruct """
recon = np.dot(proj,comp) #3088X35 * 35X155 = 3088X155
"""  MSE Error """
print("recon MSE = %.6G" %(np.mean((features - recon)**2)))
plt.figure()
plt.plot(features[:, 0], recon[:, 0], "ro")
plt.show()

trans_features = pca.fit_transform(features)
print("MSE: Transformed MSE vs projection\n", np.mean(((trans_features-proj)**2), axis= 0))
from scipy.stats import spearmanr
print("Correlation: Transformed MSE vs projection\n", spearmanr(trans_features[:,-10], proj[:, -10]))
plt.plot(trans_features[:, -10], proj[:, -10], "ro")
plt.show()

# transform input features
train_features = pca.transform(features)
test_features = pca.transform(test_features)
valid_features = pca.transform(valid_features)
all_features = pca.transform(all_features)
GS_features = pca.transform(GS_features)