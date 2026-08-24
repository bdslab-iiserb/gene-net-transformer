from convertdata import *
from utils import load_data  
import os
import torch
import numpy as np

class LoadData1( object ):

    def __init__(self, path, train_links, features_file, normalize_features=True,
                 undirected=False):
        self.path = path
        self.undirected = undirected
        self.train_links = train_links
        self.datafile = features_file

        print(path)

        self.attrfile = os.path.join(self.path, "data_standard.txt")
        if os.path.isfile(self.attrfile):
            os.remove(self.attrfile)

        data = load_data(self.datafile, normalize_features)

        data.to_csv(self.attrfile, header=None, sep=' ', mode='a')

        self.node_map = {}
        self.nodes = {}
        self.X = {}

        self.node_neighbors_map = {} 
        self.construct_nodes()
        self.read_link()
        self.construct_node_neighbors_map()
        self.construct_X()
        self.edge_index = self._create_edge_index(self.links)

    def readExp(self):
        f = open(self.attrfile)
        line = f.readline()
        items = line.strip().split(' ')
        self.attr_M = len(items[1:])
        print("Dimension of attributes:", self.attr_M)

    def construct_nodes(self):
        '''construct the dictionary '''
        self.readExp()
        f = open(self.attrfile)
        i = 0
        self.nodes['node_id'] = []
        self.nodes['node_attr'] = []
        line = f.readline()

        while line:
            line = line.strip().split(' ')

            self.node_map[int(line[0])] = i
            self.nodes['node_id'].append(i) 
            self.nodes['node_attr'].append(line[1:])
            i = i + 1
            line = f.readline()
        f.close()
        self.id_N = i
        print("Number of genes:", self.id_N)

    def construct_X(self):
        self.X['data_id_list'] = np.ndarray(shape=(len(self.links)), dtype=np.int32)
        self.X['data_attr_list'] = np.ndarray(shape=(len(self.links),  self.attr_M), dtype=np.float32)
        self.X['data_label_list'] = np.ndarray(shape=(len(self.links), 1), dtype=np.int32)
        a=self.X['data_label_list']


        #input()
        for i in range(len(self.links)):
            src_internal = int(self.node_map[int(self.links[i][0])])
            tgt_internal = int(self.node_map[int(self.links[i][1])])
            self.X['data_id_list'][i] = src_internal
            self.X['data_attr_list'][i] = self.nodes['node_attr'][src_internal]
            self.X['data_label_list'][i, 0] = tgt_internal  # one neighbor of the node



    def construct_node_neighbors_map(self):
        for link in self.links:
            if self.node_map[ int(link[0]) ] not in self.node_neighbors_map:
                self.node_neighbors_map[self.node_map[ link[0] ]] = set([self.node_map[int(link[1])]])
            else:
                self.node_neighbors_map[self.node_map[ link[0] ]].add(self.node_map[int(link[1])])

    def read_link(self):  # read link file to a list of links
        self.links = []
        if self.undirected:
            print("Making adjacency matrix symmetric since the graph is undirected.")
        else:
            print("Adjacency matrix is DIRECTED (asymmetric) as given in the data.")
        for edge in self.train_links:
            link = [int(edge[0]), int(edge[1])]
            self.links.append(link)
            if self.undirected:
                link = [int(edge[1]), int(edge[0])]
                self.links.append(link)

    def _create_edge_index(self, links):
        edge_index = torch.tensor(links, dtype=torch.long).t().contiguous()
        return edge_index
