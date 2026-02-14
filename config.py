def get_config():
    return {
        'model_id': 'Ganga-BOD-Test',
        'data_path': './data/SWD_informers.csv',
        'target': 'BOD (mg/l)',

        # Sequence lengths
        'seq_len': 24,
        'label_len': 6,
        'pred_len': 1,

        # Model dimensions
        'enc_in': 26,    # Total input features (24 input params + BOD = 25 features after dropping Month and Year)
        'dec_in': 26,    # Decoder also gets 25 inputs (as the decoder expects same dimension)
        'c_out': 1,      # Predicting 1 target (BOD)

        # Training parameters (hyperparameters for tuning)
        'batch_size': 32,  # Experiment with 16, 32, 64
        'epochs': 50,      # Experiment with 30, 50, 100 epochs
        'lr': 0.0001,      # Experiment with 0.0001, 0.0005, 0.001 for learning rate

        # Informer-specific parameters
        'factor': 5,
        'd_model': 256,    # Experiment with 128, 256, 512 for hidden dimensions
        'n_heads': 8,      # Experiment with 4, 8, 16 for number of attention heads
        'e_layers': 3,     # Experiment with 2, 3, 4 layers for encoder
        'd_layers': 2,     # Experiment with 1, 2, 3 layers for decoder
        'd_ff': 1024,      # Experiment with 512, 1024, 2048 for feed-forward dimensions
        'dropout': 0.1,    # Experiment with 0.1, 0.2, 0.3 for dropout rate
        'attn': 'prob',    # Experiment with 'prob', 'full', or 'linear' for attention types
        'embed': 'timeF',  # Use time-based embedding
        'freq': 'm',       # Monthly data frequency
        'activation': 'gelu',  # GELU is a good activation function for this model
        'output_attention': False,  # Do not output attention weights
        'distil': True,    # Use distillation for model compression

        # Others
        'features': 'M',          # Multivariate
        'target_idx': 25,         # BOD was column index 25 (0-based, total 26 cols before dropping Month, Year)
        'train_ratio': 0.7,
        'val_ratio': 0.15,
        'use_gpu': True,
        'num_workers': 0,
        'shuffle': True,
    }
