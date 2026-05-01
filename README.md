# GNT
Precise inference of gene interaction networks is essential for understanding cellular mechanisms and disease progression. Existing approaches often depend on curated databases, which may miss unknown or context-specific interactions, while many deep learning models remain database-restricted or gene-type specific.

GeneNet Transformer (GNT) addresses this gap by introducing a transformer-based deep learning framework capable of predicting missing edges in gene interaction networks. GNT integrates heterogeneous biological interaction data with gene expression profiles and leverages multi-head attention to uncover complex regulatory relationships. Through this unified and attention-driven modeling strategy, GNT delivers a robust, interpretable, and computationally efficient approach for inferring gene regulatory networks from single-cell transcriptomic data.

## Repository Structure

- `GNT_main.py` – main script for training and evaluation
- `GNT.py` – implementation of the GeneNet Transformer model
- `LoadData.py` – data loading utilities
- `convertdata.py` – preprocessing and data conversion functions
- `evaluation.py` – evaluation metrics and helper functions
- `utils.py` – utility functions for embedding construction and related tasks
- `requirements.txt` – Python dependencies
- `GNT_demo.ipynb` – example Jupyter notebooks for end-to-end execution
- `beeline_dataset/` – benchmark datasets

## Example Notebook

An end-to-end example notebook is provided in:

- `GNT_demo.ipynb`

## Parameter Description

The default GNT hyperparameters are defined as follows:

- `id_embedding_size`: Dimensionality of the gene identity embedding, which captures node-specific structural information.
- `attr_embedding_size`: Dimensionality of the attribute embedding derived from gene expression features.
- `representation_size`: Size of the final latent representation used for downstream edge modeling.
- `alpha`: Weight controlling the contribution of attribute information relative to identity information in the learned representation.
- `n_neg_samples`: Number of negative samples generated per positive edge during training.
- `epoch`: Number of training epochs used for optimizing the GNT model.
- `batch_size`: Number of samples processed in each training batch.
- `learning_rate`: Step size used by the optimizer during parameter updates.

## Default Hyperparameters

The default GNT parameters used in the example run are:

```python
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
```

## Reproducibility Notes

- Install dependencies from `requirements.txt`
- Use the provided benchmark directory structure
- The example notebook and main script reproduce the end-to-end workflow on the included dataset configuration
- For consistent results, set the same random seed used in the experiments

## 🚀 Running a Demo of GeneNet Transformer (GNT)

Follow the steps below to run a complete demo of the GNT model.

---

### 1️⃣ Clone the repository
```bash
git clone https://github.com/PranavSadhu/GNT.git
cd GNT
```

### 2️⃣ Create and activate a virtual environment
```bash
python3 -m venv gnt_env
source gnt_env/bin/activate       # Linux / macOS
# OR
gnt_env\Scripts\activate          # Windows
```
### 3️⃣ Install dependencies
```bash
pip install -r requirements.txt
```
### 4️⃣ Run the GNT model
```bash
python GNT_main.py
```
