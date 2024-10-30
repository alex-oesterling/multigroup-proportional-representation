import numpy as np 

def get_argsup(dataset, curation_set, indices, sklearn_regressor):
    k = int(np.sum(indices))
    if curation_set is not None:
        m = curation_set.shape[0]
        expanded_dataset = np.concatenate((dataset, curation_set), axis=0)
        curation_indicator = np.concatenate((np.zeros(dataset.shape[0]), np.ones(curation_set.shape[0])))
        a_expanded = np.concatenate((indices, np.zeros(curation_set.shape[0])))
        m = curation_set.shape[0]
        alpha = (a_expanded/k - curation_indicator/m)
        reg = sklearn_regressor.fit(expanded_dataset, alpha)
    else:
        m = dataset.shape[0]
        alpha = (indices/k - 1/m)
        reg = sklearn_regressor.fit(dataset, alpha)
    return reg

def mpr(dataset, curation_set, indices, sklearn_regressor, return_cx=False):
    # assert np.sum(indices) == k, "Indices must sum to k"
    m = curation_set.shape[0]
    k = np.sum(indices)

    reg = get_argsup(dataset, curation_set, indices, sklearn_regressor)
    expanded_dataset = np.concatenate((dataset, curation_set), axis=0)
    c = reg.predict(expanded_dataset)
    c /= np.linalg.norm(c)
    c *= np.sqrt(m*k/(m+k))
    mpr = np.abs(np.sum((indices/k)*c[:dataset.shape[0]]) - np.sum((1/m)*c[dataset.shape[0]:]))
    if return_cx:
        return mpr, c
    return mpr