import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, train_test_split

import LoadData1 as data_loader
from UGNT1 import UGNT1
from splitedges import load_edge_splits              
from Uevaluation1 import (
    get_edge_embeddings_concat,        
    evaluate_concat_representation,    
    evaluate_directionality,           
)

parameters = {
    'id_embedding_size'   : 128,
    'attr_embedding_size' : 128,
    'representation_size' : 128,
    'alpha'               : 1,
    'n_neg_samples'       : 10,
    'epoch'               : 30,
    'batch_size'          : 256,
    'learning_rate'       : 0.002,
}

SEED = 2018
VAL_FRAC = 0.1

USE_TARGET_ROLE = False


def train_classifier(train_data, train_labels):
    clf = RandomForestClassifier(
        class_weight='balanced',
        random_state=2018,
        n_jobs=-2,
    )
    clf_parameters = {
        'clf__n_estimators'     : [200, 500],
        'clf__max_depth'        : [None, 5, 10],
        'clf__min_samples_leaf' : [2, 4],
    }

    pipeline = Pipeline([('clf', clf)])
    grid = GridSearchCV(pipeline, clf_parameters, scoring='roc_auc', cv=5)
    grid.fit(train_data, train_labels)
    return grid


def load_and_train_model(data_dir, out_dir=None, dataset_type='', sub_folder='',
                         main_folder=''):

    if out_dir is None:
        out_dir = data_dir

    print(f"\n{'='*60}")
    print(f"Loading pre-generated directed edge splits from: {out_dir}")
    print(f"{'='*60}")
    paths = load_edge_splits(out_dir)

    gene_id_file = pd.read_csv(paths['gene_ID'], sep="\t")
    feature_file = paths['expression_values']

    train_edges       = np.load(paths['train_edges'])
    train_edges_false = np.load(paths['train_edges_false'])
    test_edges        = np.load(paths['test_edges'])
    test_edges_false  = np.load(paths['test_edges_false'])

    print(f"\n{'='*60}")
    print(f"Dataset : {dataset_type} / {sub_folder} / {main_folder}")
    print(f"{'='*60}")

    enc_train_pos, val_pos = train_test_split(
        train_edges, test_size=VAL_FRAC, random_state=SEED, shuffle=True)
    enc_train_neg, val_neg = train_test_split(
        train_edges_false, test_size=VAL_FRAC, random_state=SEED, shuffle=True)

    val_edges  = np.concatenate([val_pos, val_neg])
    val_labels = np.concatenate([np.ones(len(val_pos)), np.zeros(len(val_neg))])
    print(f"Encoder-train positives: {len(enc_train_pos)}  |  "
          f"held-out validation edges: {len(val_edges)}")

    Data  = data_loader.LoadData1(out_dir + '/',
                                 train_links=enc_train_pos,
                                 features_file=feature_file)
    model = UGNT1('', Data, 2018, parameters)

    print("\n-- Training GNT encoder --")
    embeddings, attr_embeddings = model.train(val_edges, val_labels)

    if USE_TARGET_ROLE and model.target_embeddings is not None:
        embeddings = np.concatenate([embeddings, model.target_embeddings], axis=1)
        print(f"Target-role features ENABLED → embedding is now "
              f"[source|target] = {embeddings.shape[1]} dims")

    embedding_dim = embeddings.shape[1]
    print(f"\nGene embedding shape: {embeddings.shape}  (N_genes × d={embedding_dim})")

    results_df, clf = evaluate_concat_representation(
        Embeddings         = embeddings,
        train_edges        = train_edges,
        train_edges_false  = train_edges_false,
        test_edges         = test_edges,
        test_edges_false   = test_edges_false,
        embedding_dim      = embedding_dim,
        train_classifier_fn= train_classifier,
        random_state       = SEED,
    )

    results_df['Dataset Type'] = dataset_type
    results_df['Sub Folder']   = sub_folder
    results_df['Main Folder']  = main_folder

    return results_df


def run_experiments(base_dir):

    dataset_types = ['Specific Dataset','Non-Specific Dataset','STRING Dataset']
    sub_folders   = ['hESC','hHEP', 'mDC', 'mESC', 'mHSC-E', 'mHSC-GM', 'mHSC-L']
    main_folders  = ['TFs+500', 'TFs+1000']

    all_results = []

    for dataset_type in dataset_types:
        for sub_folder in sub_folders:
            for main_folder in main_folders:
                data_dir = os.path.join(base_dir, dataset_type,
                                        sub_folder, main_folder)
                print(f'\nProcessing: {dataset_type} – {sub_folder} – {main_folder}')
                print(f'  data_dir: {os.path.abspath(data_dir)}')
                try:
                    df = load_and_train_model(
                        data_dir=data_dir,
                        out_dir=os.path.join(data_dir, 'generated'),
                        dataset_type=dataset_type,
                        sub_folder=sub_folder,
                        main_folder=main_folder,
                    )
                    all_results.append(df)
                except Exception as e:
                    import traceback
                    print(f"  ERROR: {e}")
                    traceback.print_exc()

    if all_results:
        final_df = pd.concat(all_results, ignore_index=True)

        col_order = [
            'Dataset Type', 'Sub Folder', 'Main Folder',
            'Representation', 'AUC-ROC', 'AUPR',
            'Precision', 'F1-Score',
            'Directionality Acc', 'Input Dim (-> clf)'
        ]
        final_df = final_df[[c for c in col_order if c in final_df.columns]]

        out_file = 'GNT_concat_raw_results.csv'
        final_df.to_csv(out_file, index=False)
        print(f"\nResults saved to: {out_file}")
        print(final_df.to_string(index=False))
    else:
        print("No results collected — check dataset paths.")


if __name__ == '__main__':
    base_dir = 'Benchmark Dataset'
    run_experiments(base_dir)