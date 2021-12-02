import numpy as np

log_bottom = np.log2(np.arange(2, 500))


def compute_clicks(preds, known_num, all_tids):
    tids_known = set(all_tids[:known_num])
    tids_rest = set(all_tids[known_num:])
    preds = [x for x in preds if x not in tids_known]
    preds = [(idx, x) for idx, x in enumerate(preds)]
    overlap = [x for x in preds if x[1] in tids_rest]
    if len(overlap) > 0:
        return max(0, (overlap[0][0] - 1) / 10)
    else:
        return 50


def compute_clicks_all(preds, known_num, all_tids):
    preds = [(idx, x) for idx, x in enumerate(preds)]
    overlap = [x for x in preds if x[1] in all_tids]
    if len(overlap) > 0:
        return max(0, (overlap[0][0] - 1) / 10)
    else:
        return 50


def compute_ndcg_all(preds, known_num, all_tids):
    labels = all_tids
    relevances = np.array([x in labels for x in preds])
    overlap = np.sum(relevances)
    if overlap > 0:
        dcg = relevances[0] + np.sum(relevances[1:] / log_bottom)
        if overlap == 1:
            idcg = 1
        else:
            idcg = 1 + np.sum(1 / log_bottom[:(overlap - 1)])
        ndcg = dcg / idcg
    else:
        ndcg = 0
    return ndcg


def compute_ndcg(preds, known_num, all_tids):
    tids_known = all_tids[:known_num]
    labels = set(all_tids[known_num:])

    preds = [x for x in preds if x not in tids_known]
    relevances = np.array([x in labels for x in preds])

    overlap = np.sum(relevances)
    if overlap > 0:
        dcg = relevances[0] + np.sum(relevances[1:] / log_bottom[:len(preds) - 1])
        if overlap == 1:
            idcg = 1
        else:
            idcg = 1 + np.sum(1 / log_bottom[:(overlap - 1)])
        ndcg = dcg / idcg
    else:
        ndcg = 0
    return ndcg


def strict_r_precision(preds, known_num, all_tids):
    tids_known = set(all_tids[:known_num])
    labels = set(all_tids[known_num:])
    preds = [x for x in preds if x not in tids_known]
    preds = preds[:len(labels)]
    portion = [x for x in preds if x in labels]

    portion = len(portion) / len(labels)
    return portion


def strict_r_precision_all(preds, known_num, all_tids):
    labels = all_tids
    preds = preds[:len(labels)]
    portion = [x for x in preds if x in labels]

    portion = len(portion) / len(labels)
    return portion


def r_precision(preds, known_num, all_tids):
    entries = all_tids[known_num:]
    portion = [x for x in entries if x in preds]
    portion = len(portion) / len(entries)

    return portion
