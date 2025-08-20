#!/usr/bin/env python
"""
Fine-tune Pre-trained Nicheformer Model for Downstream Tasks

This script fine-tunes a pre-trained Nicheformer model for downstream tasks 
and stores predictions in an AnnData object.
"""

import os
import argparse
import logging
from typing import Optional, Dict, Any

import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
from torch.utils.data import DataLoader
import anndata as ad

from nicheformer.models import Nicheformer
from nicheformer.models._nicheformer_fine_tune import NicheformerFineTune
from nicheformer.data.dataset import NicheformerDataset, compute_technology_mean, create_splits
from nicheformer.config_files._config_fine_tune import sweep_config as fine_tune_config
from nicheformer.config_files._config_train import sweep_config as train_config
from nicheformer.config_files._config_embeddings import sweep_config as embeddings_config


def setup_logging(log_level: str = "INFO") -> None:
    """Set up logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


def get_default_config() -> Dict[str, Any]:
    """Get default configuration parameters from config files."""
    # Start with base config from fine-tune config
    config = fine_tune_config.copy()
    
    # Add additional training parameters from train config
    config.update({
        'context_length': train_config.get('context_length', 1500),
        'learnable_pe': train_config.get('learnable_pe', True),
        'specie': train_config.get('specie', True),
        'assay': train_config.get('assay', True),
        'modality': train_config.get('modality', True),
    })
    
    # Override with paths and parameters specific to this environment
    config.update({
        'data_path': '/home/ubuntu/nicheformer/tsv2_bladder.h5ad',
        'technology_mean_path': '/home/ubuntu/nicheformer/data/model_means/dissociated_mean_script.npy',
        'checkpoint_path': '/home/ubuntu/nicheformer/nicheformer.ckpt',  # Override config file path with local path
        'output_path': 'output/predictions.h5ad',
        'output_dir': 'output/checkpoints',
        
        # Training parameters from notebook that aren't in config files
        'max_seq_len': train_config.get('context_length', 1500),
        'aux_tokens': 30,
        'chunk_size': 1000,
        'num_workers': 4,
        'precision': 32,
        'gradient_clip_val': 1.0,
        'accumulate_grad_batches': 10,
        
        # Model parameters - use config values but override some from notebook
        'extract_layers': [11],
        'function_layers': "mean",
        'dim_prediction': 33,  # From notebook
        'n_classes': 1,  # Will be updated based on data
        'reinit_layers': config.get('reinit_layers', False),
        'regress_distribution': True,  # From notebook
        'label': 'cell_type'  # From notebook
    })
    
    return config


def load_and_prepare_data(config: Dict[str, Any]) -> tuple:
    """Load data and create datasets."""
    logging.info("Loading data...")
    
    # Load data
    adata = ad.read_h5ad(config['data_path'])
    
    # Compute technology mean
    technology_mean = compute_technology_mean(adata)
    logging.info(f"Technology mean shape: {technology_mean.shape}")
    
    # Check distribution
    logging.info(f"Total cells: {len(adata):,}")
    logging.info("Cell type counts:")
    counts = adata.obs['cell_type'].value_counts()
    logging.info(f"Rare classes (< 3 cells): {(counts < 3).sum()}")
    logging.info(f"Top 10 cell types:\n{counts.head(10)}")
    
    # Create splits with filtering
    adata = create_splits(
        adata, 
        train_frac=0.7, 
        val_frac=0.15, 
        test_frac=0.15, 
        random_state=42, 
        stratify_col='cell_type',
        min_cells_per_class=3
    )
    
    # Update config with actual number of classes
    n_unique_cell_types = len(adata.obs['cell_type'].unique())
    config['n_classes'] = n_unique_cell_types
    logging.info(f"Final n_classes: {n_unique_cell_types}")
    
    return adata, technology_mean


def create_datasets_and_loaders(adata, technology_mean, config: Dict[str, Any], vocab_size: int = 20345) -> tuple:
    """Create datasets and data loaders."""
    logging.info("Creating datasets...")
    
    # Create datasets
    train_dataset = NicheformerDataset(
        adata=adata,
        technology_mean=technology_mean,
        split='train',
        max_seq_len=config['max_seq_len'],
        aux_tokens=config.get('aux_tokens', 30),
        chunk_size=config.get('chunk_size', 1000),
        metadata_fields={'obs': ['cell_type']},
        vocab_size=vocab_size
    )

    val_dataset = NicheformerDataset(
        adata=adata,
        technology_mean=technology_mean,
        split='val',
        max_seq_len=config['max_seq_len'],
        aux_tokens=config.get('aux_tokens', 30),
        chunk_size=config.get('chunk_size', 1000),
        metadata_fields={'obs': ['cell_type']},
        vocab_size=vocab_size
    )

    test_dataset = NicheformerDataset(
        adata=adata,
        technology_mean=technology_mean,
        split='test',
        max_seq_len=config['max_seq_len'],
        aux_tokens=config.get('aux_tokens', 30),
        chunk_size=config.get('chunk_size', 1000),
        metadata_fields={'obs': ['cell_type']},
        vocab_size=vocab_size
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config.get('num_workers', 4),
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config.get('num_workers', 4),
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config.get('num_workers', 4),
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


def load_model_and_create_fine_tuner(config: Dict[str, Any]):
    """Load pre-trained model and create fine-tuning model."""
    logging.info("Loading model...")
    
    # Load pre-trained model
    model = Nicheformer.load_from_checkpoint(
        checkpoint_path=config['checkpoint_path'], 
        strict=False
    )
    logging.info("Model loaded successfully")
    
    # Get vocabulary size from the model
    vocab_size = model.embeddings.weight.shape[0]
    logging.info(f"Model vocabulary size: {vocab_size}")
    
    # Create fine-tuning model
    fine_tune_model = NicheformerFineTune(
        backbone=model,
        supervised_task=config['supervised_task'],
        extract_layers=config['extract_layers'],
        function_layers=config['function_layers'],
        lr=config['lr'],
        warmup=config['warmup'],
        max_epochs=config['max_epochs'],
        dim_prediction=config['dim_prediction'],
        n_classes=config['n_classes'],
        freeze=config['freeze'],
        reinit_layers=config['reinit_layers'],
        extractor=config['extractor'],
        regress_distribution=config['regress_distribution'],
        pool=config['pool'],
        predict_density=config['predict_density'],
        ignore_zeros=config['ignore_zeros'],
        organ=config.get('organ', 'unknown'),
        label=config['label'],
        without_context=True
    )
    
    logging.info("Fine-tuning model created")
    
    # Log backbone model hyperparameters
    logging.info("Backbone model hyperparameters:")
    logging.info(f"  modality: {model.hparams.modality}")
    logging.info(f"  assay: {model.hparams.assay}")  
    logging.info(f"  specie: {model.hparams.specie}")
    
    return fine_tune_model, vocab_size


def create_trainer(config: Dict[str, Any]) -> pl.Trainer:
    """Create PyTorch Lightning trainer."""
    logging.info("Creating trainer...")
    
    # Ensure output directory exists
    os.makedirs(config['output_dir'], exist_ok=True)
    
    trainer = pl.Trainer(
        max_epochs=config['max_epochs'],
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        default_root_dir=config['output_dir'],
        precision=config.get('precision', 32),
        gradient_clip_val=config.get('gradient_clip_val', 1.0),
        accumulate_grad_batches=config.get('accumulate_grad_batches', 10),
    )
    
    logging.info("Trainer created")
    return trainer


def train_and_evaluate(trainer, fine_tune_model, train_loader, val_loader, test_loader):
    """Train and evaluate the model."""
    logging.info("Training the model...")
    
    try:
        # Train the model
        trainer.fit(
            model=fine_tune_model,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader
        )
        
        # Test the model
        logging.info("Testing the model...")
        test_results = trainer.test(
            model=fine_tune_model,
            dataloaders=test_loader
        )
        
        # Get predictions
        logging.info("Getting predictions...")
        predictions = trainer.predict(fine_tune_model, dataloaders=test_loader)
        predictions = [
            torch.cat([p[0] for p in predictions]).cpu().numpy(),
            torch.cat([p[1] for p in predictions]).cpu().numpy()
        ]
        
        return predictions, test_results
        
    except Exception as e:
        logging.error(f"Error during training: {e}")
        logging.error(f"Error type: {type(e).__name__}")
        raise


def save_results(adata, predictions, test_results, config: Dict[str, Any]) -> None:
    """Save predictions and results to AnnData object."""
    logging.info("Saving results...")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(config['output_path']), exist_ok=True)
    
    # Store predictions in AnnData object
    prediction_key = f"predictions_{config.get('label', 'X_niche_1')}"
    test_mask = adata.obs.nicheformer_split == 'test'

    if 'classification' in config['supervised_task']:
        # For classification tasks
        adata.obs.loc[test_mask, f"{prediction_key}_class"] = predictions[0]
        adata.obs.loc[test_mask, f"{prediction_key}_class_probs"] = predictions[1]
    else:
        # For regression tasks
        if 'regression' in config['supervised_task']:
            predictions = predictions[0]  # For regression both values are the same
        adata.obs.loc[test_mask, prediction_key] = predictions

    # Store test metrics
    if test_results:
        for metric_name, value in test_results[0].items():
            adata.uns[f"{prediction_key}_metrics_{metric_name}"] = value

    # Save updated AnnData
    adata.write_h5ad(config['output_path'])
    logging.info(f"Results saved to {config['output_path']}")


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Fine-tune Pre-trained Nicheformer Model for Downstream Tasks"
    )
    
    # Data paths
    parser.add_argument('--data-path', type=str, 
                       help='Path to your AnnData file')
    parser.add_argument('--technology-mean-path', type=str,
                       help='Path to technology mean file')
    parser.add_argument('--checkpoint-path', type=str,
                       help='Path to pre-trained model checkpoint')
    parser.add_argument('--output-path', type=str, default='output/predictions.h5ad',
                       help='Where to save results')
    parser.add_argument('--output-dir', type=str, default='output/checkpoints',
                       help='Directory for training checkpoints')
    
    # Training parameters
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--max-epochs', type=int, default=100,
                       help='Maximum number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of data loader workers')
    
    # Model parameters
    parser.add_argument('--supervised-task', type=str, default='niche_classification',
                       help='Type of supervised task')
    parser.add_argument('--label', type=str, default='cell_type',
                       help='Target variable to predict')
    parser.add_argument('--organ', type=str, default='brain',
                       help='Organ type')
    
    # Logging
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    
    return parser.parse_args()


def main():
    """Main execution function."""
    # Parse arguments
    args = parse_arguments()
    
    # Set up logging
    setup_logging(args.log_level)
    
    # Get default config and update with command line arguments
    config = get_default_config()
    
    # Update config with provided arguments
    for key, value in vars(args).items():
        if value is not None:
            # Convert argument names from dash to underscore
            config_key = key.replace('-', '_')
            config[config_key] = value
    
    logging.info("Configuration:")
    for key, value in config.items():
        logging.info(f"  {key}: {value}")
    
    try:
        # Set CUDA launch blocking for debugging
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
        
        # Load and prepare data
        adata, technology_mean = load_and_prepare_data(config)
        
        # Load model and create fine-tuner (need to do this before datasets to get vocab_size)
        fine_tune_model, vocab_size = load_model_and_create_fine_tuner(config)
        
        # Create datasets and loaders with correct vocab_size
        train_loader, val_loader, test_loader = create_datasets_and_loaders(
            adata, technology_mean, config, vocab_size
        )
        
        # Create trainer
        trainer = create_trainer(config)
        
        # Train and evaluate
        predictions, test_results = train_and_evaluate(
            trainer, fine_tune_model, train_loader, val_loader, test_loader
        )
        
        # Save results
        save_results(adata, predictions, test_results, config)
        
        logging.info("Fine-tuning completed successfully!")
        
    except Exception as e:
        logging.error(f"Error in main execution: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
