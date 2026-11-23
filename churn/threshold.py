"""Select an outreach cutoff from out-of-fold training probabilities."""

import numpy as np


def choose_threshold(actual, probabilities) -> float:
    labels = np.asarray(actual, dtype=int)
    scores = np.asarray(probabilities, dtype=float)
    order = np.argsort(-scores, kind="stable")
    ranked_scores = scores[order]
    # Every candidate includes a complete tie group because scoring uses >=.
    ends = np.r_[np.flatnonzero(ranked_scores[:-1] != ranked_scores[1:]), len(scores) - 1]
    cutoffs = ranked_scores[ends]
    true_positives = labels[order].cumsum()[ends]
    flagged = ends + 1
    if cutoffs[0] < 1.0:
        cutoffs = np.r_[1.0, cutoffs]
        true_positives = np.r_[0, true_positives]
        flagged = np.r_[0, flagged]
    positives = labels.sum()

    return float(cutoffs[np.argmax(2 * true_positives / (flagged + positives))])
