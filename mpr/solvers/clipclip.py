import numpy as np
from .utils import *

class ClipClip():
    # As defined in the paper "Are Gender-Neutral Queries Really Gender-Neutral? Mitigating Gender Bias in Image Search" (Wang et. al. 2021)
    def __init__(self, features, device='cuda'):
        self.features = features
        self.device = device
        self.m = features.shape[0]
        self.orderings = None

    def fit(self, k, num_cols_to_drop, query_embedding):
        if self.orderings is None:
            raise ValueError("Orderings must be provided for ClipClip, call compute_feature_MI_order first.")
        indices = self.orderings[num_cols_to_drop:]
        indices = np.sort(indices)
        clip_features = self.features[:, indices]
        clip_query = query_embedding[:, indices]
        # clip_features = torch.index_select(torch.tensor(self.features), 1, torch.tensor(self.orderings[num_cols_to_drop:].copy()).to(self.device))
        # clip_query = torch.index_select(torch.tensor(query_embedding), 1, torch.tensor(self.orderings[num_cols_to_drop:].copy()).to(self.device))

        similarities = (clip_features @ clip_query.T).flatten()
        selections = np.argsort(similarities.squeeze())[::-1][:k]
        #selections = similarities.argsort(descending=True).cpu().flatten()[:k]
        indices = np.zeros(self.m)
        indices[selections] = 1    
        AssertionError(np.sum(indices)==k)
        return indices, selections
    
    def compute_feature_MI_order(self, features, data, sensitive_attributes, n_neighbors = 10, rs=1):
        labels_arr = np.reshape(data[:,sensitive_attributes], (-1, len(sensitive_attributes))) # enable intersectional groups
        labels = np.array([' '.join(map(str, row)) for row in labels_arr])
        feature_MI = calc_feature_MI(features, labels, n_neighbors, rs)
        feature_order = np.argsort(feature_MI)[::-1]
        self.orderings = feature_order