from __future__ import print_function
import argparse
from hyperopt import hp
from hyperopt.pyll.stochastic import sample
import pickle
import sys

# Custom generator for our dataset
from src.hyperband import Hyperband
from src.train import train
from src.test import test
from src.dataset.datasetLoader import TextColor



class wrapHyperband:
    # Paramters of the model
    # depth=28 widen_factor=4 drop_rate=0.0
    def __init__(self, train_file, test_file, debug_mode, gpu_mode, num_classes):
        self.space = {
            'depth': hp.choice('depth', (28, 34, 40)),
            'widen_factor': hp.choice('widen_factor', (2, 4, 6, 8)),
            'batch_size': hp.choice('batch_size', (20, 40, 100)),
            'dropout_rate': hp.quniform('dropout', 0, 0.5, 0.1),
            'learning_rate': hp.loguniform('lr', -10, -2),
            'l2': hp.loguniform('l2', -10, -2),
        }
        self.train_file = train_file
        self.test_file = test_file
        self.debug_mode = debug_mode
        self.gpu_mode = gpu_mode
        self.num_classes = num_classes
        self.seq_len = 3
        self.iteration_jump = 1

    def get_params(self):
        return sample(self.space)


    def try_params(self, n_iterations, params):
        # Number of iterations or epoch for the model to train on
        n_iterations = int(round(n_iterations))
        print(params)
        sys.stderr.write(TextColor.BLUE + ' Loss: ' + str(n_iterations) + "\n" + TextColor.END)
        sys.stderr.write(TextColor.BLUE + str(params) + "\n" + TextColor.END)

        depth = params['depth']
        widen_factor = params['widen_factor']
        drop_rate = params['dropout_rate']
        batch_size = params['batch_size']
        epoch_limit = n_iterations
        learning_rate = params['learning_rate']
        l2 = params['l2']

        model = train(self.train_file, depth, widen_factor, drop_rate, batch_size, epoch_limit, learning_rate, l2,
                      self.debug_mode, self.gpu_mode, self.seq_len, self.iteration_jump, self.num_classes)
        stats_dictionary = test(model, self.test_file, batch_size, self.num_classes,
                                self.gpu_mode, self.seq_len, self.debug_mode)
        return stats_dictionary

    def run(self, save_output):
        hyperband = Hyperband(self.get_params, self.try_params, max_iteration=50, downsample_rate=3)
        results = hyperband.run()

        if save_output:
            with open('results.pkl', 'wb') as f:
                pickle.dump(results, f)

        # Print top 5 configs based on loss
        results = sorted(results, key=lambda r: r['loss'])[:5]
        print(results)

if __name__ == '__main__':
    '''
    Processes arguments and performs tasks to generate the pileup.
    '''
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument(
        "--train_file",
        type=str,
        required=True,
        help="Training data description csv file."
    )
    parser.add_argument(
        "--test_file",
        type=str,
        required=True,
        help="Training data description csv file."
    )
    parser.add_argument(
        "--debug_mode",
        type=bool,
        default=False,
        help="If true then debug mode is on."
    )
    parser.add_argument(
        "--gpu_mode",
        type=bool,
        default=False,
        help="If true then cuda is on."
    )
    FLAGS, unparsed = parser.parse_known_args()
    wh = wrapHyperband(FLAGS.train_file, FLAGS.test_file, FLAGS.debug_mode, FLAGS.gpu_mode, 4)
    wh.run(save_output=False)
    # train(FLAGS.train_file, FLAGS.validation_file, FLAGS.batch_size, FLAGS.epoch_size, FLAGS.model_out, FLAGS.gpu_mode)