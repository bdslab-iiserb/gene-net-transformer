import os
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score, average_precision_score # confusion_matrix, classification_report, precision_score, recall_score, f1_score
from tqdm import tqdm
import LoadData as data
from convertdata import *
from GNT import GNT
from evaluation import *

parameters = {
    'id_embedding_size': 128,
    'attr_embedding_size': 128,
    'representation_size': 128,
    'alpha': 1,
    'n_neg_samples': 10,
    'epoch': 30,
    'batch_size': 256,
    'learning_rate': 0.002
}

# Define the classifier and parameters for GridSearchCV
def train_classifier(train_data, train_labels):
    clf = SVC(class_weight='balanced', probability=True)
    clf_parameters = {
        'clf__C': [ 1, 90]
    }
    
    pipeline = Pipeline([('clf', clf)])
    grid = GridSearchCV(pipeline, clf_parameters, scoring='roc_auc', cv=10)
    grid.fit(train_data, train_labels)
    
    return grid

# Load datasets and train model
def load_and_train_model(base_dir, dataset_type, sub_folder, main_folder):
    directory_path = os.path.join(base_dir, dataset_type, sub_folder, main_folder)
    gene_id_file = pd.read_csv(os.path.join(directory_path, 'gene_ID.tsv'), sep="\t")

    feature_file = os.path.join(directory_path, 'expression_values.csv')
    link_file = os.path.join(directory_path, "edgelist.csv")
    
    # Load datasets
    train_edges = np.load(os.path.join(directory_path, 'train_edges.npy'))
    train_edges_false = np.load(os.path.join(directory_path, 'train_edges_false.npy'))
    test_edges = np.load(os.path.join(directory_path, 'test_edges.npy'))
    test_edges_false = np.load(os.path.join(directory_path, 'test_edges_false.npy'))

    print('#####')

    Data = data.LoadData(directory_path +'/', train_links=train_edges, features_file=feature_file)
    model = GNT('', Data, 2018, parameters)
    training_edges = np.concatenate([train_edges, train_edges_false])
    train_edge_labels = np.concatenate([np.ones(len(train_edges)), np.zeros(len(train_edges_false))])
    
    # Train the GNE model
    embeddings, attr_embeddings = model.train(training_edges, train_edge_labels)

    # Create a dictionary mapping gene IDs to embeddings
    gene_ids = gene_id_file['GeneName']  # Adjust the column name based on your data
    gene_embeddings_dict = {gene_ids[i]: embeddings[i] for i in range(len(gene_ids))}

    # Get edge embeddings
    pos_train_edge_embs = get_edge_embeddings(embeddings, train_edges)
    neg_train_edge_embs = get_edge_embeddings(embeddings, train_edges_false)
    train_edge_embs = np.concatenate([pos_train_edge_embs, neg_train_edge_embs])
    train_edge_labels = np.concatenate([np.ones(len(train_edges)), np.zeros(len(train_edges_false))])

    # Shuffle training data
    index = np.random.permutation(len(train_edge_labels))
    train_data = train_edge_embs[index, :]
    train_labels = train_edge_labels[index]

    # Train classifier using SVC with GridSearchCV
    grid = train_classifier(train_data, train_labels)

    # Test classifier
    pos_test_edge_embs = get_edge_embeddings(embeddings, test_edges)
    neg_test_edge_embs = get_edge_embeddings(embeddings, test_edges_false)
    test_edge_embs = np.concatenate([pos_test_edge_embs, neg_test_edge_embs])

    test_edge_labels = np.concatenate([np.ones(len(test_edges)), np.zeros(len(test_edges_false))])
    test_preds = grid.predict_proba(test_edge_embs)[:, 1]

    # Evaluate model
    test_roc = roc_auc_score(test_edge_labels, test_preds)
    test_ap = average_precision_score(test_edge_labels, test_preds)
    
    return test_roc, test_ap

# Loop through all dataset types, main folders, and subfolders
def run_experiments(base_dir):
    sub_folders = ['hESC'] # 'hHEP', 'mDC', 'mESC', 'mHSC-E', 'mHSC-GM', 'mHSC-L']
    dataset_types = ['Specific Dataset' ]  #'Non-Specific Dataset', 'Specific Dataset','STRING Dataset'
    main_folders = ['TFs+1000'] #'TFs+500'

    results = []

    for dataset_type in dataset_types:
        for sub_folder in sub_folders:
            for main_folder in main_folders:
                print(f'Processing: {dataset_type} - {sub_folder} - {main_folder}')
                test_roc, test_ap = load_and_train_model(base_dir, dataset_type, sub_folder, main_folder)
                results.append({
                    'Dataset Type': dataset_type,
                    'Sub Folder': sub_folder,
                    'Main Folder': main_folder,
                    'AUC-ROC': test_roc,
                    'AUPR': test_ap
                })

    # Save results to a CSV file
    # results_df = pd.DataFrame(results)
    # results_df.to_csv('GNT_specific_results.csv', index=False)
    print(results)

# Example of running experiments
base_dir = 'beeline_dataset/Benchmark Dataset'
run_experiments(base_dir)
