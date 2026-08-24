import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def _read_csv_first_col_index(path): 
    df = pd.read_csv(path)
    first = df.columns[0]
    if first.startswith("Unnamed") or first == "":
        df = df.drop(columns=[first])
    return df


def build_expression_file(expr_csv, target_df, out_path):
    expr = pd.read_csv(expr_csv, index_col=0)         
    ordered = target_df.sort_values("index")
    ordered = ordered[ordered["Gene"].isin(expr.index)]
    expr = expr.loc[ordered["Gene"].values]           
    expr.index = ordered["index"].values               
    expr_T = expr.T                                     
    expr_T.columns = expr.index
    expr_T.to_csv(out_path, sep="\t", index=False)
    return expr_T.shape                                 


def sample_negative_edges(pos_edges, tf_indices, gene_indices, n_needed, seed=2018):
    rng = np.random.default_rng(seed)
    pos_set = set(map(tuple, pos_edges.tolist()))
    tf_indices = np.asarray(tf_indices)
    gene_indices = np.asarray(gene_indices)

    neg_set = set()
    while len(neg_set) < n_needed:
        batch = max(n_needed - len(neg_set), 1) * 2
        s = rng.choice(tf_indices, size=batch)
        t = rng.choice(gene_indices, size=batch)
        for a, b in zip(s.tolist(), t.tolist()):
            if a == b:
                continue
            pair = (a, b)
            if pair in pos_set or pair in neg_set:
                continue
            neg_set.add(pair)
            if len(neg_set) >= n_needed:
                break
    return np.array(sorted(neg_set), dtype=int)


def edge_split_paths(out_dir):

    return {
        "train_edges":       os.path.join(out_dir, "train_edges.npy"),
        "train_edges_false": os.path.join(out_dir, "train_edges_false.npy"),
        "val_edges":         os.path.join(out_dir, "val_edges.npy"),
        "val_edges_false":   os.path.join(out_dir, "val_edges_false.npy"),
        "test_edges":        os.path.join(out_dir, "test_edges.npy"),
        "test_edges_false":  os.path.join(out_dir, "test_edges_false.npy"),
        "expression_values": os.path.join(out_dir, "expression_values.csv"),
        "gene_ID":           os.path.join(out_dir, "gene_ID.tsv"),
    }


def load_edge_splits(out_dir):
    paths = edge_split_paths(out_dir)
    missing = [k for k, p in paths.items() if not os.path.isfile(p)]
    if missing:
        raise FileNotFoundError(
            f"Generated files not found in '{os.path.abspath(out_dir)}':\n    "
            + "\n    ".join(paths[k] for k in missing)
            + "\n\nRun  `python generate_edges2.py`  first to create them for "
              "this dataset case."
        )
    return paths


def _three_way_split(edges, test_size, val_size, seed):
    trainval, test = train_test_split(
        edges, test_size=test_size, random_state=seed, shuffle=True)
    val_rel = val_size / (1.0 - test_size)
    train, val = train_test_split(
        trainval, test_size=val_rel, random_state=seed, shuffle=True)
    return train, val, test


def generate_edge_splits(data_dir, out_dir=None, test_size=0.1, val_size=0.1,
                         neg_ratio=1, seed=2018):
    if out_dir is None:
        out_dir = data_dir
    os.makedirs(out_dir, exist_ok=True)

    required = ["Label.csv", "TF.csv", "Target.csv", "BL--ExpressionData.csv"]
    missing  = [f for f in required
                if not os.path.isfile(os.path.join(data_dir, f))]
    if missing:
        raise FileNotFoundError(
            "Missing data file(s) in "
            f"'{os.path.abspath(data_dir)}':\n    "
            + "\n    ".join(missing)
            + "\n\nPut these files in that folder, or point `base_dir` at the "
              "folder that contains them."
        )

    label_df  = _read_csv_first_col_index(os.path.join(data_dir, "Label.csv"))
    tf_df     = _read_csv_first_col_index(os.path.join(data_dir, "TF.csv"))
    target_df = _read_csv_first_col_index(os.path.join(data_dir, "Target.csv"))

    pos_edges = label_df[["TF", "Target"]].to_numpy().astype(int)
    pos_edges = np.unique(pos_edges, axis=0)

    tf_indices   = tf_df["index"].to_numpy().astype(int)
    gene_indices = target_df["index"].to_numpy().astype(int)
    n_genes      = int(gene_indices.max()) + 1

    print(f"Genes: {n_genes}  |  TFs: {len(tf_indices)}  |  "
          f"positive edges: {len(pos_edges)}")

    n_neg = len(pos_edges) * neg_ratio
    neg_edges = sample_negative_edges(pos_edges, tf_indices, gene_indices,
                                      n_neg, seed=seed)
    print(f"Sampled {len(neg_edges)} negative edges")
    assert set(map(tuple, pos_edges.tolist())).isdisjoint(
        set(map(tuple, neg_edges.tolist()))), "positive/negative overlap!"

    train_edges, val_edges, test_edges = _three_way_split(
        pos_edges, test_size, val_size, seed)
    train_edges_false, val_edges_false, test_edges_false = _three_way_split(
        neg_edges, test_size, val_size, seed)

    S = lambda a: set(map(tuple, a.tolist()))
    for A, B in [(train_edges, val_edges), (train_edges, test_edges),
                 (val_edges, test_edges),
                 (train_edges_false, val_edges_false),
                 (train_edges_false, test_edges_false),
                 (val_edges_false, test_edges_false)]:
        assert S(A).isdisjoint(S(B)), "train/val/test overlap!"

    paths = edge_split_paths(out_dir)
    np.save(paths["train_edges"],       train_edges)
    np.save(paths["train_edges_false"], train_edges_false)
    np.save(paths["val_edges"],         val_edges)
    np.save(paths["val_edges_false"],   val_edges_false)
    np.save(paths["test_edges"],        test_edges)
    np.save(paths["test_edges_false"],  test_edges_false)

    n_exp, n_g = build_expression_file(
        os.path.join(data_dir, "BL--ExpressionData.csv"), target_df,
        paths["expression_values"])
    print(f"expression_values.csv written: {n_exp} experiments x {n_g} genes")

    target_df[["Gene", "index"]].to_csv(paths["gene_ID"], sep="\t", index=False)

    print("\nGenerated splits (train : val : test = "
          f"{int(round((1-test_size-val_size)*100))} : "
          f"{int(round(val_size*100))} : {int(round(test_size*100))}):")
    print(f"  train_edges  : {train_edges.shape}   train_edges_false  : {train_edges_false.shape}")
    print(f"  val_edges    : {val_edges.shape}   val_edges_false    : {val_edges_false.shape}")
    print(f"  test_edges   : {test_edges.shape}   test_edges_false   : {test_edges_false.shape}")

    return paths



def run_generation(base_dir, dataset_types, sub_folders, main_folders,
                   test_size=0.1, val_size=0.1, neg_ratio=1, seed=2018):
    """
    For every (dataset_type, sub_folder, main_folder) combination, read the raw
    data from  base_dir / <dataset_type> / <sub_folder> / <main_folder>  and
    write the generated splits into that case's  'generated/'  sub-folder.
    """
    n_ok = 0
    for dataset_type in dataset_types:
        for sub_folder in sub_folders:
            for main_folder in main_folders:
                data_dir = os.path.join(base_dir, dataset_type,
                                        sub_folder, main_folder)
                out_dir  = os.path.join(data_dir, "generated")
                print(f"\n{'='*70}")
                print(f"Generating: {dataset_type} / {sub_folder} / {main_folder}")
                print(f"  raw data : {os.path.abspath(data_dir)}")
                print(f"  saving to: {os.path.abspath(out_dir)}")
                print(f"{'='*70}")
                try:
                    generate_edge_splits(data_dir, out_dir=out_dir,
                                         test_size=test_size, val_size=val_size,
                                         neg_ratio=neg_ratio, seed=seed)
                    n_ok += 1
                except FileNotFoundError as e:
                    print(f"  SKIP: {e}")
    print(f"\nDone. Generated splits for {n_ok} case(s).")


if __name__ == "__main__":
    base_dir      = "Benchmark Dataset1"
    dataset_types = ["Specific Dataset", "Non-Specific Dataset", "STRING Dataset"]   
    sub_folders   = ["hESC", "hHEP", "mDC", "mESC", "mHSC-E", "mHSC-GM", "mHSC-L"] 
    main_folders  = ["TFs+500", "TFs+1000"]          # add: "TFs+500"

    TEST_SIZE = 0.1
    VAL_SIZE  = 0.1
    NEG_RATIO = 1
    SEED      = 2018

    run_generation(base_dir, dataset_types, sub_folders, main_folders,
                   test_size=TEST_SIZE, val_size=VAL_SIZE,
                   neg_ratio=NEG_RATIO, seed=SEED)
