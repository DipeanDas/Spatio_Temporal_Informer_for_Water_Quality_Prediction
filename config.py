def get_config():
    return {
        'model_id': 'BOD-Test',
        'data_path': 'put dataset url',
        'target': 'BOD (mg/l)',

        # Sequence lengths
        'seq_len': 24,
        'label_len': 6,
        'pred_len': 1,

        # Model dimensions
        'enc_in': 26,    
        'dec_in': 26,    
        'c_out': 1,      

        # Training parameters (hyperparameters for tuning)
        'batch_size': 32,  
        'epochs': 50,      
        'lr': 0.0001,      

        # Informer-specific parameters
        'factor': 5,
        'd_model': 256,    
        'n_heads': 8,      
        'e_layers': 3,     
        'd_layers': 2,     
        'd_ff': 1024,      
        'dropout': 0.1,    
        'attn': 'prob',    
        'embed': 'timeF',  
        'freq': 'm',       
        'activation': 'gelu',  
        'output_attention': False,  
        'distil': True,    

        # Others
        'features': 'M',          
        'target_idx': 25,         
        'train_ratio': 0.7,
        'val_ratio': 0.15,
        'use_gpu': True,
        'num_workers': 0,
        'shuffle': True,
    }

