import numpy as np
from sklearn.metrics import (average_precision_score, roc_auc_score,
                             precision_score, f1_score)
import pandas as pd


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def evaluate_ROC_from_matrix(X_edges, y_true, matrix):

    y_predict = [sigmoid(matrix[int(edge[0]), int(edge[1])]) for edge in X_edges]
    roc = roc_auc_score(y_true, y_predict)
    pr = average_precision_score(y_true, y_predict)
    return roc, pr


def load_embedding(embedding_file, N, combineAttribute=False, datafile=None):
    f = open(embedding_file)
    line = f.readline().strip().split(' ')
    d = int(line[1])
    embeddings = np.random.randn(int(N), d)
    line = f.readline()
    while line:
        line = line.strip().split(' ')
        embeddings[int(line[0]), :] = line[1:]
        line = f.readline()
    f.close()
    if combineAttribute:
        data = load_datafile(datafile, N)
        embeddings = np.hstack((embeddings, data))
    return embeddings


def load_datafile(data_file, N):
    f = open(data_file)
    line = f.readline().strip().split(' ')
    d = len(line[1:])
    data = np.zeros([int(N), d])
    i = 0
    while line:
        data[int(line[0]), :] = line[1:]
        i += 1
        line = f.readline()
        if i < N:
            line = line.strip().split(' ')
        else:
            break
    f.close()
    return data


def get_edge_embeddings_concat(Embeddings, edge_list):
    edge_array = np.array(edge_list)
    src = Embeddings[edge_array[:, 0].astype(int)]
    tgt = Embeddings[edge_array[:, 1].astype(int)]
    return np.concatenate([src, tgt], axis=1)


def evaluate_directionality(clf, Embeddings, directed_edges, device=None):
    directed_edges = np.array(directed_edges)
    reversed_edges = directed_edges[:, ::-1].copy()

    scores_fwd = clf.predict_proba(
        get_edge_embeddings_concat(Embeddings, directed_edges))[:, 1]
    scores_rev = clf.predict_proba(
        get_edge_embeddings_concat(Embeddings, reversed_edges))[:, 1]

    correct = (scores_fwd > scores_rev).sum()
    acc = correct / len(directed_edges)
    print(f"  Directionality: {acc:.4f}  ({correct}/{len(directed_edges)} correct)")
    return float(acc)


def evaluate_concat_representation(Embeddings, train_edges, train_edges_false,
                                   test_edges, test_edges_false,
                                   embedding_dim,
                                   train_classifier_fn,
                                   device=None,
                                   random_state=None):
    train_labels = np.concatenate([
        np.ones(len(train_edges)), np.zeros(len(train_edges_false))])
    test_labels = np.concatenate([
        np.ones(len(test_edges)), np.zeros(len(test_edges_false))])

    print("\n-- Concat-raw -> classifier  (asymmetric, no MLP, 2d features) --")
    cat_tr = np.concatenate([
        get_edge_embeddings_concat(Embeddings, train_edges),
        get_edge_embeddings_concat(Embeddings, train_edges_false)])
    cat_te = np.concatenate([
        get_edge_embeddings_concat(Embeddings, test_edges),
        get_edge_embeddings_concat(Embeddings, test_edges_false)])

    rng = np.random.default_rng(random_state)
    idx = rng.permutation(len(train_labels))
    clf = train_classifier_fn(cat_tr[idx], train_labels[idx])
    preds = clf.predict_proba(cat_te)[:, 1]

    roc = roc_auc_score(test_labels, preds)
    pr = average_precision_score(test_labels, preds)
    bin_ = (preds >= 0.5).astype(int)
    prec = precision_score(test_labels, bin_, zero_division=0)
    f1 = f1_score(test_labels, bin_, zero_division=0)
    dire = evaluate_directionality(clf, Embeddings, test_edges)

    results_df = pd.DataFrame([{
        'Representation': 'Concat-raw -> classifier',
        'AUC-ROC': round(roc, 4),
        'AUPR': round(pr, 4),
        'Precision': round(prec, 4),
        'F1-Score': round(f1, 4),
        'Directionality Acc': round(dire, 4),
        'Input Dim (-> clf)': 2 * embedding_dim,
    }])

    print("\n" + "=" * 90)
    print("EDGE REPRESENTATION RESULT  (Concat-raw)")
    print("=" * 90)
    print(results_df.to_string(index=False))
    print("=" * 90)

    return results_df, clf