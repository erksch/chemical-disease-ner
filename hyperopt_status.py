import pickle
import hyperopt

trials = pickle.load(open('./trials.p', 'rb'))
print(f"{len(trials.trials)} trials executed.")
print('Best model so far:')
print(trials.best_trial)
