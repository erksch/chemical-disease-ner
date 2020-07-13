import os
from configparser import ConfigParser
from argparse import ArgumentParser

def load_config():
    configparser = ConfigParser()
    argparser = ArgumentParser()
    argparser.add_argument('--config', required=True, help='Path to config file')
    args, _ = argparser.parse_known_args()

    if not os.path.exists(args.config):
        raise Exception('Configuration file {} does not exist'.format(args.config))
    
    print(f"Loading config from file {args.config}.")
    configparser.read(args.config)

    CONFIG = {}

    # network
    CONFIG['hidden_dim'] = configparser.getint('network', 'hidden_dim')
    CONFIG['dropout'] = configparser.getfloat('network', 'dropout')
    CONFIG['use_dropout'] = configparser.getboolean('network', 'use_dropout')

    # embeddings
    CONFIG['use_pretrained_embeddings'] = configparser.getboolean('embeddings', 'use_pretrained_embeddings')

    if CONFIG['use_pretrained_embeddings']:
        CONFIG['pretrained_embeddings_path'] = configparser.get('embeddings', 'pretrained_embeddings_path')
    else:
        CONFIG['embeddings_dim'] = configparser.getint('embeddings', 'embeddings_dim')
        
    # training
    CONFIG['epochs'] = configparser.getint('training', 'epochs')
    CONFIG['learning_rate'] = configparser.getfloat('training', 'learning_rate')
    CONFIG['momentum'] = configparser.getfloat('training', 'momentum')

    # batching
    CONFIG['batch_mode'] = configparser.get('batching', 'batch_mode')
    if CONFIG['batch_mode'] == 'padded_sentences':
        CONFIG['padded_sentences_batch_size'] = configparser.getint('batching', 'padded_sentences_batch_size') 
    
    return CONFIG