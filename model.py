"""
Random Forest from Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - impurity
def impurity(labels):
    """Return a non-negative impurity score for a 1D array of integer class labels."""
    # score how mixed the labels are; 0 for a pure set, larger for more mixed sets.

    if len(labels) == 0:
        return 0.0
    
    _, counts = np.unique(labels, return_counts=True)
    p = counts / counts.sum()
    
    return float(1 - np.sum(p ** 2))

# Step 2 - split_dataset
import numpy as np

def split_dataset(features, labels, feature_index, threshold):
    # partition rows into left (feature <= threshold) and right (feature > threshold)
    
    mask = features[:, feature_index] <= threshold

    left_features = features[mask]
    left_labels = labels[mask]

    right_features = features[~mask]
    right_labels = labels[~mask]

    return left_features, left_labels, right_features, right_labels

# Step 3 - split_score
def split_score(parent_labels, left_labels, right_labels):
    # TODO: return a score where higher means the children are purer than the parent.

    n = len(parent_labels)
    w_l = len(left_labels) / n
    w_r = len(right_labels) / n

    return impurity(parent_labels) - (w_l * impurity(left_labels) + w_r * impurity(right_labels))

# Step 4 - best_split
import numpy as np

def best_split(features, labels, feature_indices):
    # search feature_indices for the (feature, threshold) that best improves purity.

    best = {
        'feature_index': None, 
        'threshold': None, 
        'score': 0.0
    }

    for j in feature_indices:

        feature = features[:, j]
        unique_values = np.unique(feature)

        # almeno due valori distinti per creare uno split
        if len(unique_values) < 2:
            continue

        thresholds = (unique_values[:-1] + unique_values[1:]) / 2

        for t in thresholds:

            _, ll, _, rl = split_dataset(features, labels, j, t)

            # skip split invalidi
            if len(ll) == 0 or len(rl) == 0:
                continue

            s = split_score(labels, ll, rl)

            if s > best["score"]:
                best["feature_index"] = j
                best["threshold"] = t
                best["score"] = s

    return best

# Step 5 - should_stop
def should_stop(labels, depth, max_depth, min_samples_split):
    """Return True if this node should become a leaf instead of splitting further."""
    # TODO: decide whether to stop growing based on purity, depth, and size...

    # Check purity
    if len(np.unique(labels)) == 1:
        return True

    # Check Depth cap
    if depth >= max_depth:
        return True

    # Check size floor
    if len(labels) < min_samples_split:
        return True

    return False

# Step 6 - leaf_prediction
def leaf_prediction(labels):
    # choose a single class label to output for a leaf given the labels that reached it
    
    classes, counts = np.unique(labels, return_counts=True)
    max_idx = np.argmax(counts)

    return int(classes[max_idx])

# Step 7 - build_tree
def build_tree(features, labels, max_depth=10, min_samples_split=2, feature_subset=None, depth=0):
    # recursively grow a decision tree, returning a nested dict of leaf/internal nodes.
    
    # seleziono le colonne
    if feature_subset is None:
        feature_subset = range(features.shape[1])

    # stopping criteria
    if should_stop(labels, depth, max_depth, min_samples_split):
        return {
            'leaf': True, 
            'prediction': leaf_prediction(labels)
        }

    # cerco il migliore split
    best = best_split(features, labels, feature_subset)

    if best["feature_index"] == None:
        return {
            'leaf': True, 
            'prediction': leaf_prediction(labels)
        }

    j = best["feature_index"] 
    t = best["threshold"]

    # partiziono dati
    lf, ll, rf, rl = split_dataset(features, labels, j, t)

    # creo i figli
    left_tree = build_tree(
        lf, ll,
        max_depth,  
        min_samples_split, 
        feature_subset, 
        depth + 1
    )

    right_tree = build_tree(
        rf, rl,
        max_depth,  
        min_samples_split, 
        feature_subset, 
        depth + 1
    )
    
    return {
        'leaf': False,
        'feature_index': j,
        'threshold': t,
        'left': left_tree,
        'right': right_tree
    }

# Step 8 - predict_example_tree (not yet solved)
# TODO: implement

# Step 9 - predict_tree (not yet solved)
# TODO: implement

# Step 10 - bootstrap_sample (not yet solved)
# TODO: implement

# Step 11 - feature_subset (not yet solved)
# TODO: implement

# Step 12 - train_forest (not yet solved)
# TODO: implement

# Step 13 - combine_predictions (not yet solved)
# TODO: implement

# Step 14 - predict_forest (not yet solved)
# TODO: implement

# Step 15 - accuracy (not yet solved)
# TODO: implement

