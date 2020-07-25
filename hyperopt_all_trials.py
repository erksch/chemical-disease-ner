import hyperopt
import pickle

trials = pickle.load(open('./trials.p', 'rb'))
for trial in trials.trials:
    print(f"trial id: {trial['tid']}")
    print(trial['result'])
    print(trial['misc']['vals'])
