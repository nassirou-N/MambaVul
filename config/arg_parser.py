import argparse

def parameter_parser():
    # Experiment parameters
    parser = argparse.ArgumentParser(description='Smart Contract Vulnerability Detection Using Mamba Neural Network')

    parser.add_argument('filename', type=str, help="name of file to process")
    parser.add_argument('-vt', type=str, choices=['ts', 're', 'io'], help="vulnerability type (ts=timestamp, re=reentrancy, io=integer overflow)")
        
    # Model hyperparameters
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('-d', '--dropout', type=float, default=0.1, help='dropout rate')
    parser.add_argument('--vec_length', type=int, default=150, help='vector dimension for Word2Vec embeddings')
    parser.add_argument('--epochs', type=int, default=30, help='number of training epochs')
    parser.add_argument('-b', '--batch_size', type=int, default=32, help='batch size')
    
    # Mamba-specific parameters
    parser.add_argument('--d_model', type=int, default=150, help='Mamba model dimension (should match vec_length)')
    parser.add_argument('--d_state', type=int, default=16, help='Mamba state dimension')
    parser.add_argument('--d_conv', type=int, default=4, help='Mamba convolution kernel size')
    parser.add_argument('--expand_factor', type=int, default=2, help='Mamba expansion factor')
    parser.add_argument('--num_mamba_layers', type=int, default=2, help='Number of Mamba blocks')
    
    # Training parameters
    parser.add_argument('--validation_split', type=float, default=0.1, help='Validation split ratio')
    parser.add_argument('--early_stopping', action='store_true', help='Enable early stopping')
    parser.add_argument('--patience', type=int, default=5, help='Early stopping patience')
    
    return parser.parse_args()