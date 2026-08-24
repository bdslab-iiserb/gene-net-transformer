import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.sparse import coo_matrix


def convertAdjMatrixToSortedRankTSV(inputFile=None, outputFilename=None, desc=True):
    tbl = inputFile


    rownames = range(tbl.shape[0])
    firstCol = np.repeat(rownames, tbl.shape[1]).reshape((tbl.shape[0]*tbl.shape[1], 1))
    secondCol = []

    x = np.array([i for i in range(tbl.shape[0])])
    secondCol = np.tile(x, len(range(tbl.shape[1])))

    secondCol = secondCol.reshape((tbl.shape[0]*tbl.shape[1], 1))
    thirdCol = np.matrix.flatten(np.matrix(tbl)).reshape((tbl.shape[0]*tbl.shape[1], 1))
    thirdCol = np.nan_to_num(thirdCol)

    result = np.column_stack((firstCol, secondCol, thirdCol))
    result = pd.DataFrame(result)
    result.columns = ['c1','c2', 'c3']

    result = pd.DataFrame(result[result['c1']!=result['c2']])
    result =  result.sort_values(['c3', 'c1', 'c2'], ascending=[0, 1, 1])
    result[['c1', 'c2']] = result[['c1', 'c2']].astype(int)
    return (result)


def convertSortedRankTSVToAdjMatrix (input=None, nodes=None, undirected=False):
    print("Converting to Adjacency matrix (directed)" if not undirected
          else "Converting to Adjacency matrix (symmetric)")
    tbl = pd.DataFrame(input).drop_duplicates()
    tbl.columns = ['c1', 'c2', 'c3']
    tbl = tbl[tbl['c3']==1]

    tbl = tbl.sort_values(['c2'], ascending=1)
    tbl[['c1', 'c2', 'c3']] = tbl[['c1', 'c2', 'c3']].astype(int)
    m = np.zeros((nodes, nodes))
    dups = tbl['c2'].duplicated()


    startIndices = list(np.where(dups== False)[0])

    for i in range(len(startIndices)-1):
        # print(i)
        colIndex = tbl.iloc[startIndices[i], 1]
        if startIndices[i]==(startIndices[i + 1] - 1):
            rowIndexes = tbl.iloc[startIndices[i], 0]
            valuesToAdd = tbl.iloc[startIndices[i], 2]
        else:
            rowIndexes = tbl.iloc[startIndices[i]:(startIndices[i + 1]), 0].values
            valuesToAdd = tbl.iloc[startIndices[i]:(startIndices[i + 1] ), 2].values
        m[rowIndexes, colIndex] = valuesToAdd
        if undirected:
            m[colIndex, rowIndexes] = valuesToAdd


    colIndex = tbl.iloc[startIndices[len(startIndices)-1], 1]
    rowIndexes = tbl.iloc[startIndices[len(startIndices)-1]:len(tbl.iloc[:, 1]), 0]
    valuesToAdd = tbl.iloc[startIndices[len(startIndices)-1]:len(tbl.iloc[:, 1]), 2]

    m[rowIndexes, colIndex] = valuesToAdd
    if undirected:
        m[colIndex, rowIndexes] = valuesToAdd

    m = pd.DataFrame(m)
    return (m)

def load_network(filename, num_genes):
    print ("### Loading [%s]..." % (filename))
    i, j, val = np.loadtxt(filename).T
    A = coo_matrix((val, (i, j)), shape=(num_genes, num_genes))
    A = A.todense()
    A = np.squeeze(np.asarray(A))
    A = A - np.diag(np.diag(A))
    return A
