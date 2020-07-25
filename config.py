import os
from configparser import ConfigParser
from argparse import ArgumentParser

def decorate_hyperparams(hyperparams):
    decorated_params =  {
        **hyperparams, 
        **hyperparams['hyperopt_use_dropout'],
        **hyperparams['hyperopt_use_pretrained_embeddings'],
        **hyperparams['hyperopt_use_weighted_loss']

    }
    del decorated_params['hyperopt_use_dropout']
    del decorated_params['hyperopt_use_pretrained_embeddings']
    del decorated_params['hyperopt_use_weighted_loss']

    decorated_params['hidden_dim'] = int(decorated_params['hidden_dim'])
    if 'embeddings_dim' in decorated_params:
        decorated_params['embeddings_dim'] = int(decorated_params['embeddings_dim'])
    decorated_params['epochs'] = int(decorated_params['epochs'])
    decorated_params['padded_sentences_batch_size'] = int(decorated_params['padded_sentences_batch_size'])

    decorated_params['optimize_hyperparameters'] = True
    
    return decorated_params


def load_config(hyperparams={}):
    configparser = ConfigParser()
    argparser = ArgumentParser()
    argparser.add_argument('--config', required=True, help='Path to config file')
    args, _ = argparser.parse_known_args()

    if not os.path.exists(args.config):
        raise Exception('Configuration file {} does not exist'.format(args.config))
    
    print(f"Loading config from file {args.config}.")
    configparser.read(args.config)

    CONFIG = {}

    #hyperopt
    if configparser.getboolean('hyperopt', 'optimize_hyperparameters'):
        decorated_params =  decorate_hyperparams(hyperparams)
        print('CONFIG:')
        print(decorated_params)
        return decorated_params
    else:
        CONFIG['optimize_hyperparameters'] = configparser.getboolean('hyperopt', 'optimize_hyperparameters')

    # data
    CONFIG['train_set_path'] = configparser.get('data', 'train_set')
    CONFIG['dev_set_path'] = configparser.get('data', 'dev_set')
    CONFIG['test_set_path'] = configparser.get('data', 'test_set')

    # network
    CONFIG['hidden_dim'] = configparser.getint('network', 'hidden_dim')
    CONFIG['dropout'] = configparser.getfloat('network', 'dropout')
    CONFIG['use_dropout'] = configparser.getboolean('network', 'use_dropout')
    CONFIG['use_additional_linear_layers'] = configparser.getboolean('network', 'use_additional_linear_layers')

    # embeddings
    CONFIG['use_pretrained_embeddings'] = configparser.getboolean('embeddings', 'use_pretrained_embeddings')

    if CONFIG['use_pretrained_embeddings']:
        CONFIG['pretrained_embeddings_path'] = configparser.get('embeddings', 'pretrained_embeddings_path')
    else:
        CONFIG['embeddings_dim'] = configparser.getint('embeddings', 'embeddings_dim')
        
    # training
    CONFIG['epochs'] = configparser.getint('training', 'epochs')
    CONFIG['optimizer'] = configparser.get('training', 'optimizer')
    CONFIG['learning_rate'] = configparser.getfloat('training', 'learning_rate')
    CONFIG['momentum'] = configparser.getfloat('training', 'sgd_momentum')
    CONFIG['use_weighted_loss'] = configparser.getboolean('training', 'use_weighted_loss')
    CONFIG['null_class_weight'] = configparser.getfloat('training', 'null_class_weight')
    CONFIG['non_null_class_weight'] = configparser.getfloat('training', 'non_null_class_weight')

    # batching
    CONFIG['batch_mode'] = configparser.get('batching', 'batch_mode')
    if CONFIG['batch_mode'] == 'padded_sentences':
        CONFIG['padded_sentences_batch_size'] = configparser.getint('batching', 'padded_sentences_batch_size') 
        CONFIG['padded_sentences_max_length'] = configparser.getint('batching', 'padded_sentences_max_length')

    # evaluation
    CONFIG['evaluation_interval'] = configparser.getint('evaluation', 'interval')
    CONFIG['evaluate_only_at_end'] = configparser.getboolean('evaluation', 'only_at_end')
    
    return CONFIG
