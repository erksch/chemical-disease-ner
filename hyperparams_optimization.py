import os.path
import pickle
from hyperopt import hp, fmin, rand, tpe, space_eval, tpe, Trials
from main import main

hyperparameter_space = {
    'train_set_path': 'data/CDR_Data/CDR.Corpus.v010516/CDR_TrainingSet.BioC.xml',
    'dev_set_path': 'data/CDR_Data/CDR.Corpus.v010516/CDR_DevelopmentSet.BioC.xml',
    'test_set_path': 'data/CDR_Data/CDR.Corpus.v010516/CDR_TestSet.BioC.xml',

    'hidden_dim': hp.quniform('hidden_dim', 1, 400, 10),
    'hyperopt_use_dropout': hp.choice('hyperopt_use_dropout', [
        {'dropout': 0.0, 'use_dropout': False},
        {'dropout': hp.uniform('dropout', 0.0, 1.0), 'use_dropout': True}
    ]),
    'use_additional_linear_layers': hp.choice('use_additional_linear_layers', [True, False]),

    'hyperopt_use_pretrained_embeddings': hp.choice('hyperopt_use_pretrained_embeddings', [
        {'use_pretrained_embeddings': False, 'embeddings_dim': hp.quniform('embeddings_dim', 1, 1000, 1)},
        {'use_pretrained_embeddings': True, 'pretrained_embeddings_path': 'embeddings/BioWordVec_PubMed_MIMICIII_d200.vec.bin'}
    ]),

    'epochs': hp.quniform('epochs', 1, 1000, 10),
    'learning_rate': hp.uniform('learning_rate', 0.00001, 0.1),
    'momentum': hp.uniform('momentum', 0.0, 1.0),
    'optimizer': hp.choice('optimizer', ['sgd', 'adam']),

    'hyperopt_use_weighted_loss': hp.choice('hyperopt_use_weighted_loss', [
        {
            'use_weighted_loss': True, 
            'null_class_weight': hp.uniform('null_class_weight', 0.0, 20.0), 
            'non_null_class_weight': hp.uniform('non_null_class_weight', 0.0, 20.0)
        },
        {
            'use_weighted_loss': False, 
            'null_class_weight': 1.0, 
            'non_null_class_weight': 1.0
        }
    ]),

    'batch_mode': 'padded_sentences',
    'padded_sentences_batch_size': hp.quniform('padded_sentences_batch_size', 10, 500, 10),
    'padded_sentences_max_length': hp.quniform('padded_sentences_batch_size', 10, 100),

    'evaluate_only_at_end': True
}

def optimize_hyperparams():
    if os.path.isfile('./trials.p'):
        trials = pickle.load(open('./trials.p', 'rb'))
    else:
        trials = Trials()

    for i in range(100):
        best_model = fmin(
            fn=main,
            space=hyperparameter_space,
            algo=tpe.suggest,
            trials=trials,
            max_evals=len(trials.trials)+1
        )
        pickle.dump(trials, open('./trials.p', 'wb'))

    print('best model:')
    print(best_model)
    return best_model

if __name__ == '__main__':
    optimize_hyperparams()
