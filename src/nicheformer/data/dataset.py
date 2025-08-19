import gc
import numpy as np
import torch
from anndata import AnnData
from scipy.sparse import issparse
from tqdm import tqdm
from torch.utils.data import Dataset
from sklearn.utils import sparsefuncs
import numba


def sf_normalize(X):
    """Normalize the input matrix to a scale of 10000."""
    X = X.copy()
    # Ensure X is float type to avoid casting errors during multiplication
    if not issparse(X) and not np.issubdtype(X.dtype, np.floating):
        X = X.astype(np.float64)
    
    counts = np.array(X.sum(axis=1))
    # avoid zero devision error
    counts += counts == 0.
    # normalize to 10000. counts
    scaling_factor = 10000. / counts

    if issparse(X):
        sparsefuncs.inplace_row_scale(X, scaling_factor)
    else:
        np.multiply(X, scaling_factor.reshape((-1, 1)), out=X)

    return X


def compute_technology_mean(adata):
    """
    Compute technology mean from AnnData object.
    
    Args:
        adata: AnnData object with gene expression data
        
    Returns:
        np.array: Technology mean values for each gene
    """
    if issparse(adata.X):
        tech_mean = np.array(adata.X.mean(axis=0)).flatten()
    else:
        tech_mean = adata.X.mean(axis=0)
    
    # Handle zeros and NaN values
    tech_mean = np.nan_to_num(tech_mean)
    tech_mean += tech_mean == 0
    
    return tech_mean


def create_splits(adata, train_frac=0.7, val_frac=0.15, test_frac=0.15, random_state=42, stratify_col=None, min_cells_per_class=3):
    """
    Create train/validation/test splits for AnnData object.
    
    Args:
        adata: AnnData object to create splits for
        train_frac: Fraction of data for training (default: 0.7)
        val_frac: Fraction of data for validation (default: 0.15)
        test_frac: Fraction of data for testing (default: 0.15)
        random_state: Random seed for reproducibility (default: 42)
        stratify_col: Column name in adata.obs to stratify splits by (optional)
        min_cells_per_class: Minimum cells per class for stratification (default: 3)
        
    Returns:
        AnnData: Modified adata object with 'nicheformer_split' column added
    """
    import pandas as pd
    from sklearn.model_selection import train_test_split
    
    # Verify fractions sum to 1
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6, "Fractions must sum to 1.0"
    
    # Filter out rare classes if stratifying
    if stratify_col is not None and stratify_col in adata.obs.columns:
        # Check class distribution
        class_counts = adata.obs[stratify_col].value_counts()
        rare_classes = class_counts[class_counts < min_cells_per_class]
        
        if len(rare_classes) > 0:
            print(f"Warning: Found {len(rare_classes)} cell types with < {min_cells_per_class} cells:")
            for cell_type, count in rare_classes.items():
                print(f"  - {cell_type}: {count} cells")
            
            # Filter out rare classes
            print(f"Removing {rare_classes.sum()} cells from rare classes...")
            mask = ~adata.obs[stratify_col].isin(rare_classes.index)
            adata = adata[mask].copy()
            print(f"Remaining cells: {len(adata):,}")
            
            # Update class counts
            class_counts = adata.obs[stratify_col].value_counts()
            print(f"Remaining cell types: {len(class_counts)}")
    
    n_cells = len(adata)
    indices = np.arange(n_cells)
    
    # Stratification data
    stratify_data = None
    if stratify_col is not None:
        if stratify_col not in adata.obs.columns:
            print(f"Warning: Stratify column '{stratify_col}' not found, using random splits")
        else:
            stratify_data = adata.obs[stratify_col].values
    
    # First split: separate train from (val + test)
    train_indices, temp_indices = train_test_split(
        indices, 
        test_size=(val_frac + test_frac),
        random_state=random_state,
        stratify=stratify_data
    )
    
    # Second split: separate val from test
    if stratify_data is not None:
        temp_stratify = stratify_data[temp_indices]
    else:
        temp_stratify = None
        
    val_indices, test_indices = train_test_split(
        temp_indices,
        test_size=(test_frac / (val_frac + test_frac)),
        random_state=random_state,
        stratify=temp_stratify
    )
    
    # Create split labels
    split_labels = np.full(n_cells, '', dtype=object)
    split_labels[train_indices] = 'train'
    split_labels[val_indices] = 'val'
    split_labels[test_indices] = 'test'
    
    # Add to adata
    adata.obs['nicheformer_split'] = split_labels
    
    # Print split statistics
    print(f"Created splits:")
    print(f"  Train: {len(train_indices):,} cells ({len(train_indices)/n_cells:.1%})")
    print(f"  Val:   {len(val_indices):,} cells ({len(val_indices)/n_cells:.1%})")
    print(f"  Test:  {len(test_indices):,} cells ({len(test_indices)/n_cells:.1%})")
    
    if stratify_col is not None and stratify_col in adata.obs.columns:
        print(f"\nSplit distribution by '{stratify_col}':")
        split_stats = pd.crosstab(adata.obs['nicheformer_split'], adata.obs[stratify_col], margins=True)
        print(split_stats)
    
    return adata


@numba.jit(nopython=True, nogil=True)
def _sub_tokenize_data(x: np.array, max_seq_len: int = -1, aux_tokens: int = 30):
    """Tokenize the input gene vector"""
    scores_final = np.empty((x.shape[0], max_seq_len if max_seq_len > 0 else x.shape[1]))
    for i, cell in enumerate(x):
        nonzero_mask = np.nonzero(cell)[0]
        sorted_indices = nonzero_mask[np.argsort(-cell[nonzero_mask])][:max_seq_len] 
        sorted_indices = sorted_indices + aux_tokens # we reserve some tokens for padding etc (just in case)
        if max_seq_len:
            scores = np.zeros(max_seq_len, dtype=np.int32)
        else:
            scores = np.zeros_like(cell, dtype=np.int32)
        scores[:len(sorted_indices)] = sorted_indices.astype(np.int32)

        scores_final[i, :] = scores

    return scores_final


def tokenize_data(x: np.array, median_counts_per_gene: np.array, max_seq_len: int = None):
    """Tokenize the input gene vector to a vector of 32-bit integers."""
    x = np.nan_to_num(x) # is NaN values, fill with 0s
    x = sf_normalize(x)
    median_counts_per_gene += median_counts_per_gene == 0
    out = x / median_counts_per_gene.reshape((1, -1))

    scores_final = _sub_tokenize_data(out, 4096, 30)

    return scores_final.astype('i4')


class NicheformerDataset(Dataset):
    """Dataset for Nicheformer"""

    def __init__(self, adata, technology_mean, split='train', max_seq_len=4096, aux_tokens=30, chunk_size=1000,
                 metadata_fields=None):
        """
        Initialize the dataset

        Args:
            adata (AnnData): Annotated data matrix
            technology_mean (np.array): technology mean
            split (str): 'train', 'test', or 'val'
            max_seq_len (int): Maximum sequence length for tokenization
            aux_tokens (int): Number of reserved tokens
            chunk_size (int): Number of cells to process at once
            metadata_fields (dict): Dictionary specifying which metadata fields to include.
                                  Format: {
                                      'obs': ['field1', 'field2'],  # fields from adata.obs
                                      'obsm': ['field3', 'field4']  # fields from adata.obsm
                                  }
        """
        self.adata = adata.copy()
        
        # Filter by split if specified
        if split is not None:
            split_column = 'nicheformer_split'
            if split_column not in self.adata.obs.columns:
                raise ValueError(f"Split column '{split_column}' not found in adata.obs. "
                               f"Please create splits first using create_splits() function.")
            
            split_mask = self.adata.obs[split_column] == split
            if not split_mask.any():
                raise ValueError(f"No cells found for split '{split}'. "
                               f"Available splits: {self.adata.obs[split_column].unique()}")
            
            self.adata = self.adata[split_mask].copy()
            print(f"Using {split} split with {len(self.adata)} cells")
        
        self.technology_mean = technology_mean
        self.max_seq_len = max_seq_len
        self.aux_tokens = aux_tokens
        self.chunk_size = chunk_size
        self.metadata_fields = metadata_fields or {'obs': [], 'obsm': []}

        # Initialize storage for tokenized data
        self.n_cells = len(self.adata)
        self.tokens = None

        # Process data in chunks
        self._process_chunks()

        # Store metadata
        self._prepare_metadata()

    def _process_chunks(self):
        n_chunks = (self.n_cells + self.chunk_size - 1) // self.chunk_size
        tokens_list = []

        for chunk_idx in tqdm(range(n_chunks)):
            start_idx = chunk_idx * self.chunk_size
            end_idx = min((chunk_idx + 1) * self.chunk_size, self.n_cells)

            # Process chunk
            chunk_tokens = self._process_chunk(start_idx, end_idx)
            tokens_list.append(chunk_tokens)

            torch.cuda.empty_cache()
            gc.collect()  # force garbage collection

        # Concatenate all chunks
        self.tokens = np.concatenate(tokens_list, axis=0)

    def _process_chunk(self, start_idx, end_idx):
        # Get chunk of data
        chunk_adata = self.adata[start_idx:end_idx]

        # Convert sparse to dense for this chunk only
        if issparse(chunk_adata.X):
            x = chunk_adata.X.toarray()
        else:
            x = chunk_adata.X

        # Process chunk
        x = np.nan_to_num(x)
        x = sf_normalize(x)

        tech_mean = self.technology_mean
        tech_mean += tech_mean == 0

        x = x / tech_mean.reshape((1, -1))

        # Tokenize
        tokens = _sub_tokenize_data(x, self.max_seq_len, self.aux_tokens).astype(np.int32)

        return tokens

    def _prepare_metadata(self):
        self.metadata = {}

        # Process obs fields - direct assignment, no special processing needed
        for field in self.metadata_fields.get('obs', []):
            self.metadata[field] = self.adata.obs[field].values

        # Process obsm fields - chunk processing only if sparse
        for field in self.metadata_fields.get('obsm', []):
            if issparse(self.adata.obsm[field]):
                vectors = []
                for chunk_idx in range(0, self.n_cells, self.chunk_size):
                    end_idx = min(chunk_idx + self.chunk_size, self.n_cells)
                    chunk = self.adata.obsm[field][chunk_idx:end_idx].toarray()
                    vectors.append(chunk)
                self.metadata[field] = np.concatenate(vectors, axis=0)
            else:
                self.metadata[field] = self.adata.obsm[field]

    def __len__(self):
        return self.n_cells

    def __getitem__(self, idx):
        item = {
            'X': torch.tensor(self.tokens[idx])
        }

        # Add all metadata fields to the item
        for key, value in self.metadata.items():
            # Handle different data types appropriately
            if isinstance(value[idx], (str, np.str_)):
                # For categorical/string data, we need to convert to numerical labels
                if not hasattr(self, '_label_encoders'):
                    self._label_encoders = {}
                
                if key not in self._label_encoders:
                    from sklearn.preprocessing import LabelEncoder
                    self._label_encoders[key] = LabelEncoder()
                    # Fit on all unique values for this field
                    unique_vals = np.unique(value)
                    self._label_encoders[key].fit(unique_vals)
                
                # Transform the current value
                encoded_val = self._label_encoders[key].transform([value[idx]])[0]
                item[key] = torch.tensor(encoded_val, dtype=torch.long)
            else:
                # For numerical data, convert to appropriate tensor type
                if np.issubdtype(type(value[idx]), np.integer):
                    item[key] = torch.tensor(value[idx], dtype=torch.long)
                elif np.issubdtype(type(value[idx]), np.floating):
                    item[key] = torch.tensor(value[idx], dtype=torch.float32)
                else:
                    item[key] = torch.tensor(value[idx])

        return item
