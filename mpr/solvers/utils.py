import numpy as np 
import sklearn.feature_selection as fs
import cvxpy as cp

def MMR_cost(indices, embeddings):
    """
    Computes the MMR cost for a given subset of embeddings.
    The MMR cost is a measure of the diversity of the subset relative to the entire set of embeddings.
    It calculates the average pairwise distance between elements in the subset, normalized by the mean
    and standard deviation of the pairwise distances in the entire set.
    Parameters:
    indices (numpy.ndarray): A binary array indicating which elements are in the subset (1) and which are not (0).
    embeddings (numpy.ndarray): A 2D array where each row is an embedding vector.
    Returns:
    float: The MMR cost, representing the normalized diversity score of the subset.
    """
    # compute the diversity score of the entire set
    marginal_diversity = 0
    subset = embeddings[indices==1]
    mean_embedding, std_embedding = statEmbedding(embeddings)
    for i in range(len(subset)):
        for j in range(i+1, len(subset)):
            marginal_diversity += (np.linalg.norm(subset[i] - subset[j])-mean_embedding)/std_embedding
    marginal_diversity /= len(subset)
    return marginal_diversity

def statEmbedding(embeddings):
    distances = []
    for i in range(len(embeddings)):
        for j in range(i+1, len(embeddings)):
            # print(embeddings[i], embeddings[j])
            distance = np.linalg.norm(embeddings[i] - embeddings[j])
            distances.append(distance)
    distances = np.array(distances)
    mean_embedding = np.mean(distances)
    std_embedding = np.std(distances)
    return mean_embedding, std_embedding

# for clipclip benchmark method
def calc_feature_MI(features, labels, n_neighbors = 10, rs=1):
    return fs.mutual_info_classif(features, labels, discrete_features=False, copy=True, n_neighbors=n_neighbors, random_state=rs)

def return_feature_MI_order(features, data, sensitive_attributes, n_neighbors = 10, rs=1):
    labels_arr = np.reshape(data[:,sensitive_attributes], (-1, len(sensitive_attributes))) # enable intersectional groups
    labels = np.array([' '.join(map(str, row)) for row in labels_arr])
    feature_MI = calc_feature_MI(features, labels, n_neighbors, rs)
    feature_order = np.argsort(feature_MI)[::-1]
    return feature_order

def get_lower_upper_bounds(k, similarity_scores, labels, curation_labels):
    if curation_labels is None:
        curation_labels = labels
    a = cp.Variable(labels.shape[0])
    curation_mean = np.mean(curation_labels, axis=0)
    
    objective = cp.Maximize(similarity_scores.T @ a)
    constraints = [(labels.T @ a)/k == curation_mean, sum(a)==k, 0<=a, a<=1]
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.ECOS)
    lb_opt = objective.value

    lb_indices = a.value


    #  find global optimal
    objective = cp.Maximize(similarity_scores.T @ a)
    constraints = [sum(a)==k, 0<=a, a<=1]
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.ECOS)
    global_opt = objective.value

    ub_indices = a.value

    return lb_indices, ub_indices

# for PBM
def fon(l):
    try:
        return l[0]
    except:
        return None
