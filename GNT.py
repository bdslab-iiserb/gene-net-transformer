import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import random
import os
from sklearn.base import BaseEstimator, TransformerMixin
import evaluation
import torch.nn.functional as F

class GNT(BaseEstimator, TransformerMixin):
    def __init__(self, path, data, random_seed=2018, parameters=None):
        self.path = path
        self.nodes = data.nodes
        self.node_neighbors_map = data.node_neighbors_map
        self.node_N = data.id_N
        self.attr_M = data.attr_M
        self.X_train = data.X
        
        self.id_embedding_size = parameters['id_embedding_size']
        self.attr_embedding_size = parameters['attr_embedding_size']
        self.batch_size = parameters['batch_size']
        self.alpha = parameters['alpha']
        self.n_neg_samples = parameters['n_neg_samples']
        self.epoch = parameters['epoch']
        self.random_seed = random_seed
        self.learning_rate = parameters['learning_rate']
        self.representation_size = parameters['representation_size']

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._init_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        print(parameters)

    def _init_model(self):
        self.model = GNTModel(self.node_N, self.attr_M, self.id_embedding_size, 
                              self.attr_embedding_size, self.representation_size, self.alpha)
        self.model = self.model.to(self.device)

    def partial_fit(self, X):
        self.model.train()
        self.optimizer.zero_grad()

        batch_data_id = torch.tensor(X['batch_data_id'], dtype=torch.long).to(self.device)
        batch_data_attr = torch.tensor(X['batch_data_attr'], dtype=torch.float32).to(self.device)
        batch_data_label = torch.tensor(X['batch_data_label'], dtype=torch.long).to(self.device).view(-1)

        # Check if any target value is out of bounds
        if torch.max(batch_data_label) >= self.node_N:
            print("Error: Target value out of bounds.")
            print("Max target:", torch.max(batch_data_label))
            print("Number of classes (node_N):", self.node_N)
            raise ValueError("Target value out of bounds")

        outputs = self.model(batch_data_id, batch_data_attr)
        loss = self.criterion(outputs, batch_data_label)
        loss.backward()
        self.optimizer.step()
        return loss.item()


    def train(self, validation_edges, validation_labels):
        best_validation_accuracy = 0.0
        last_improvement = 0
        require_improvement = 2

        print('Using structure and attribute embedding')
        for epoch in range(self.epoch):
            random.seed(epoch)
            perm = np.random.permutation(len(self.X_train['data_id_list']))
            self.X_train['data_id_list'] = self.X_train['data_id_list'][perm]
            self.X_train['data_attr_list'] = self.X_train['data_attr_list'][perm]
            self.X_train['data_label_list'] = self.X_train['data_label_list'][perm]

            total_batch = int(len(self.X_train['data_id_list']) / self.batch_size)
            avg_cost = 0.
            for i in range(total_batch):
                batch_xs = {}
                random.seed(epoch * i)
                start_index = np.random.randint(0, len(self.X_train['data_id_list']) - self.batch_size)
                batch_xs['batch_data_id'] = self.X_train['data_id_list'][start_index:(start_index + self.batch_size)]
                batch_xs['batch_data_attr'] = self.X_train['data_attr_list'][start_index:(start_index + self.batch_size)]
                batch_xs['batch_data_label'] = self.X_train['data_label_list'][start_index:(start_index + self.batch_size)]

                cost = self.partial_fit(batch_xs)
                avg_cost += cost / total_batch

            Embeddings_out = self.getEmbedding('out_embedding', self.nodes)
            Embeddings_in = self.getEmbedding('embed_layer', self.nodes)
            Embeddings = Embeddings_out + Embeddings_in

            adj_matrix_rec = np.dot(Embeddings, Embeddings.T)
            roc, pr = evaluation.evaluate_ROC_from_matrix(validation_edges, validation_labels, adj_matrix_rec)

            if roc > best_validation_accuracy:
                best_validation_accuracy = roc
                last_improvement = epoch
                self.embedding_checkpoints(Embeddings, "save", "all")
                improved_str = '*'
            else:
                improved_str = ''

            msg = "Epoch: {0:>6}, Train-Batch Loss: {1:.9f}, Validation AUC: {2:.9f} {3}"
            print(msg.format(epoch + 1, avg_cost, roc, improved_str))

            if epoch - last_improvement > require_improvement:
                print("No improvement found in a while, stopping optimization.")
                break

        Embeddings = self.embedding_checkpoints(Embeddings, "restore", "all")
        attr_embeddings = self.getEmbedding('attribute', self.nodes)
        return Embeddings, attr_embeddings
    
    def getEmbedding(self, embedding_type, nodes):
        self.model.eval()
        with torch.no_grad():
            nodes_df = pd.DataFrame(nodes)
            nodes = {'node_id': nodes_df['node_id'].tolist(),
                     'node_attr': nodes_df['node_attr'].tolist()}
            
            # Debugging: Print the shape and type of node attributes before tensor conversion
            # print(f"node_attr type: {type(nodes['node_attr'])}, length: {len(nodes['node_attr'])}")
            # for i, attr in enumerate(nodes['node_attr']):
            #     print(f"node_attr[{i}] type: {type(attr)}, length: {len(attr)}, value: {attr[:5]}")
            #     if i >= 5:  # Only print the first 5 elements to avoid too much output
            #         break
            nodes['node_attr'] = [[float(value) for value in attr] for attr in nodes['node_attr']]

            node_id = torch.tensor(nodes['node_id'], dtype=torch.long).to(self.device)
            node_attr = torch.tensor(nodes['node_attr'], dtype=torch.float32).to(self.device)

            if embedding_type == 'embed_layer':
                embeddings = self.model.get_representation(node_id, node_attr)
            elif embedding_type == 'out_embedding':
                embeddings = self.model.out_embeddings.weight.cpu().numpy()
            elif embedding_type == 'attribute':
                embeddings = self.model.attr_embeddings.weight.cpu().numpy()
            elif embedding_type == 'structure':
                embeddings = self.model.in_embeddings.weight.cpu().numpy()
            return embeddings

    def embedding_checkpoints(self, Embeddings, type, embedding_type="all"):
        file = self.path + "Embeddings_" + embedding_type + ".txt"
        if type == "save":
            if os.path.isfile(file):
                os.remove(file)
            pd.DataFrame(Embeddings).to_csv(file, index=False, header=False)
        elif type == 'restore':
            Embeddings = pd.read_csv(file, header=None)
            return np.array(Embeddings)

class GNTModel(nn.Module):
    def __init__(self, node_N, attr_M, id_embedding_size, attr_embedding_size, representation_size, alpha, num_transformer_layers=2, num_heads=4):
        super(GNTModel, self).__init__()
        self.in_embeddings = nn.Embedding(node_N, id_embedding_size)
        self.attr_embeddings = nn.Linear(attr_M, attr_embedding_size)
        self.hidden_layer = nn.Linear(id_embedding_size + attr_embedding_size, representation_size)
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=representation_size,
            nhead=num_heads
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.transformer_encoder_layer,
            num_layers=num_transformer_layers
        )
        self.out_embeddings = nn.Linear(representation_size, node_N)  # Ensure output matches node_N
        self.alpha = alpha

    def forward(self, node_id, node_attr):
        id_embed = F.normalize(self.in_embeddings(node_id), dim=1)
        attr_embed = F.normalize(F.elu(self.attr_embeddings(node_attr)), dim=1)
        
        embed_layer = torch.cat([id_embed, self.alpha * attr_embed], dim=1)
        representation_layer = torch.tanh(self.hidden_layer(embed_layer))
        representation_layer = representation_layer.unsqueeze(0)  # Add a batch dimension for transformer
        transformed_layer = self.transformer_encoder(representation_layer)
        transformed_layer = transformed_layer.squeeze(0)  # Remove the added batch dimension
        return self.out_embeddings(transformed_layer)

    def get_representation(self, node_id, node_attr):
        with torch.no_grad():
            id_embed = F.normalize(self.in_embeddings(node_id), dim=1)
            attr_embed = F.normalize(F.elu(self.attr_embeddings(node_attr)), dim=1)
            embed_layer = torch.cat([id_embed, self.alpha * attr_embed], dim=1)
            representation_layer = torch.tanh(self.hidden_layer(embed_layer))
            representation_layer = representation_layer.unsqueeze(0)  # Add a batch dimension for transformer
            transformed_layer = self.transformer_encoder(representation_layer)
            transformed_layer = transformed_layer.squeeze(0)  # Remove the added batch dimension
        return transformed_layer.cpu().numpy()
