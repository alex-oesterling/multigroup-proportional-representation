from .utils import statEmbedding
import numpy as np

class MMR():
    def __init__(self, retreival_embeddings, similarity_scores):
        self.m = similarity_scores.shape[0]
        self.similarity_scores = similarity_scores
        # define what embedding to use for the diversity metric.
        # None (img itself), CLIP, or CLIP+PCA
        self.embeddings = retreival_embeddings # the entire dataset (also used for retrieval), or a separate curation set
        self.mean_embedding = None
        self.std_embedding = None

    def fit(self, k, lambda_):
        if self.mean_embedding is None or self.std_embedding is None:
            self.mean_embedding, self.std_embedding =  statEmbedding(self.embeddings)

        indices = np.zeros(self.m)
        selection = []
        for i in range(k):
            MMR_temp = np.full(len(self.embeddings), -np.inf)
            if i==0:
                idx = np.argmax(self.similarity_scores)
                selection.append(idx)
                indices[idx] = 1
                continue
            for j in range(len(self.embeddings)):
                if indices[j] == 1:
                    continue
                # temporary select the jth element
                indices[j] = 1
                score_sim = (self.similarity_scores.T @ indices - self.mean_embedding)/self.std_embedding
                score_diversity = self.marginal_diversity_score(indices,j)
                MMR_temp[j] = (1-lambda_)* score_sim + lambda_ * score_diversity
                indices[j] = 0
            # select the element with the highest MMR 
            idx = np.argmax(MMR_temp)
            selection.append(idx)
            indices[np.argmax(MMR_temp)] = 1
        assert np.sum(indices)==k, "The number of selected items is not equal to k"
        MMR_cost = self.marginal_diversity_score(indices)
        return indices, MMR_cost, selection

    def marginal_diversity_score(self, indices, addition_index=None):
        # compute the diversity score of the entire set
        if addition_index is None:
            marginal_diversity = 0
            subset = self.embeddings[indices==1]
            for i in range(len(subset)):
                for j in range(i+1, len(subset)):
                    marginal_diversity += (np.linalg.norm(subset[i] - subset[j])-self.mean_embedding)/self.std_embedding
            marginal_diversity /= len(subset)

        # compute the diversity score of an additional item
        else:
            # compute the diversity score of adding the addition_index to the current set
            marginal_diversity = 0
            indices[addition_index] = 0
            subset = self.embeddings[indices==1]
            for i in range(len(subset)):
                marginal_diversity += (np.linalg.norm(self.embeddings[addition_index] - subset[i])-self.mean_embedding)/self.std_embedding
            marginal_diversity /= len(subset)
        return marginal_diversity