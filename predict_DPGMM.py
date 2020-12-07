import pickle

import numpy as np
import pandas as pd
from sklearn.mixture import BayesianGaussianMixture
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

with open('DPGMM_model.pkl', 'rb') as f:
    trained_model = pickle.load(f)
with open('PCA_weights.pkl', 'rb') as f:
    pca_weights = pickle.load(f)
with open('cluster_idx.pkl', 'rb') as f:
    cluster_idx = pickle.load(f)
#Replace the values of the pandas dataframe "random_testset" with real scaled data of the corresponding features
random_testset = pd.read_csv('random_testset.csv', index_col='index')

random_testset = np.matmul(random_testset.to_numpy(), np.transpose(pca_weights))

predicted_prob = trained_model.predict_proba(random_testset)[0]
predicted_label = np.argmax([predicted_prob[cluster_idx[0]], predicted_prob[cluster_idx[1]],
                            predicted_prob[cluster_idx[2]], predicted_prob[cluster_idx[3]],
                            predicted_prob[cluster_idx[4]], predicted_prob[cluster_idx[5]]])

print("The predicted assignment probabilities are:")
print("C1: %2f, C2: %2f, C3: %2f, C4: %2f, C5: %2f, C6: %2f" % (predicted_prob[cluster_idx[0]],
                                                                predicted_prob[cluster_idx[1]],
                                                                predicted_prob[cluster_idx[2]],
                                                                predicted_prob[cluster_idx[3]],
                                                                predicted_prob[cluster_idx[4]],
                                                                predicted_prob[cluster_idx[5]]))
print("Predicted cluster: C%d" % (predicted_label+1))
