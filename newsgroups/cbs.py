import numpy as np
from sklearn.feature_selection import chi2
from sklearn import preprocessing


def center(p_vectors, n_vectors, alpha=10, beta=4):
    p_nor = preprocessing.normalize(p_vectors)
    n_nor = preprocessing.normalize(n_vectors)
    return alpha*np.mean(p_nor, axis=0) - beta*np.mean(n_nor, axis=0)


def sim_cos(vecs, c):
    vecs_nor = preprocessing.normalize(vecs)
    c_nor = preprocessing.normalize(c[None, :])[0]
    return vecs_nor @ c_nor


def sim_gow(vecs, c):
    vecs_nor = preprocessing.normalize(vecs)
    c_nor = preprocessing.normalize(c[None, :])[0]
    return 1 - np.mean(np.abs(vecs_nor-c_nor), axis=1)


def sim_lor(vecs, c):
    return 1 - np.sum(np.log(1+np.abs(vecs-c)), axis=1)


def sim_dice(vecs, c, epsilon=1e-8):
    vecs_sum = np.sum(vecs**2, axis=1)
    c_sum = np.sum(c**2)
    return 2*vecs@c/(vecs_sum+c_sum+epsilon)


def sim_jac(vecs, c, epsilon=1e-8):
    vecs_sum = np.sum(vecs**2, axis=1)
    c_sum = np.sum(c**2)
    vecs_c = vecs @ c
    return vecs_c/(vecs_sum+c_sum-vecs_c+epsilon)


def sims(vecs, c, epsilon=1e-8):
    sims_cos = sim_cos(vecs, c)
    sims_gow = sim_gow(vecs, c)
    sims_lor = sim_lor(vecs, c)
    sims_dice = sim_dice(vecs, c, epsilon)
    sims_jac = sim_jac(vecs, c, epsilon)
    return np.vstack([sims_cos, sims_gow, sims_lor, sims_dice, sims_jac]).T


def generate_cbs_features(p_set, n_set, *sets,
                          n_select_features=600, alpha=10, beta=4):
    labels = np.concatenate(
        [np.ones(p_set.shape[0]), -np.ones(n_set.shape[0])])
    chi_score = chi2(np.concatenate([p_set, n_set]), labels)
    s_idxs = np.argsort(-chi_score[0])[:n_select_features]
    p_vectors = p_set[:, s_idxs]
    n_vectors = n_set[:, s_idxs]
    c = center(p_vectors, n_vectors, alpha, beta)
    cbs_features = [sims(p_vectors, c), sims(n_vectors, c)]
    for se in sets:
        cbs_features.append(sims(se[:, s_idxs], c))
    return cbs_features


def generate_cbs_features_multi(
        p_set, n_set, *sets,
        n_select_features=600, alpha=10, beta=4):
    cbs_features = []
    for i in range(p_set.shape[1]):
        cbs_features.append(generate_cbs_features(
            p_set[:, i, :], n_set[:, i, :], *[se[:, i, :] for se in sets],
            n_select_features=n_select_features, alpha=alpha, beta=beta))
    res = []
    for i in range(len(cbs_features[0])):
        res.append(
            np.hstack([cbs_feature[i] for cbs_feature in cbs_features]))
    return res
