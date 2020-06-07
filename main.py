import argparse
import os
import sys

import pandas as pd

import utils
from TAC import transductive_clustering
from baselines import knn_classifier, transductive_svm_multiclass, transductive_svm, baselines
from commons import Commons


def Run(game, algorithm, features='cs', num_loops=5000, train_size=0.9):
    logger = utils.get_logger()
    logger.info("Running parameters: ")
    logger.info("   Game: %s", game)
    logger.info("   Algorithm: %s", algorithm)
    logger.info("   Features: %s", features)
    logger.info("   Num loops: %d", num_loops)
    logger.info("   Train set size: %0.2f", train_size)

    utils.verify_folders()
    df_traits = utils.get_trait_df(features)
    # preprocessing, for instance scaling
    df_traits = utils.preprocess_traits(df_traits)
    df_game = utils.get_game_data(game)
    # Merge the features with the game data
    df_traits = pd.merge(df_game, df_traits, on='orig_id')
    # x- features, y- label
    x = df_traits.drop(['orig_id', game], axis=1)
    y = df_traits[game]

    results = None
    # Run algorithm
    if algorithm in ['tac', 'knn', 'mvc', 'erg', 'ewg']:
        # When running TAC or KNN --> need to convert the game column from categorical type to numeric
        labels_list = list(y.astype('category').cat.categories)
        labels_dict = dict(zip(range(len(labels_list)), labels_list))
        y = y.astype('category').cat.codes

        if algorithm == 'tac':
            results = transductive_clustering(x, y, game, labels_dict, num_loops, train_size)
        elif algorithm == 'knn':
            results = knn_classifier(x, y, game, labels_dict, num_loops, train_size)
        else:
            results = baselines(y, algorithm, labels_dict)

    elif algorithm == 'svm':
        if game == 'door':
            results = transductive_svm_multiclass(x, y, game, num_loops, train_size)
        else:
            results = transductive_svm(x, y, game, num_loops, train_size)

    if results is None:
        raise Exception("main.Run:Could not retrieve results.")
    results.to_csv(os.path.join(Commons.RESULTS_PATH, algorithm + '_' + str(features) + '_' + game + '.csv'),
                   index=False)
    logger.info("Running results: " + str(results))

    return results


if __name__ == '__main__':
    try:
        parser = argparse.ArgumentParser(add_help=True)
        parser.add_argument('-g',
                            help='Choose the game you want to run: chicken, box, or door (default is chicken).',
                            default='chicken')
        parser.add_argument('-a',
                            help='Choose the model you want to run: tac, knn, or svm (default is tac).',
                            default='tac')
        parser.add_argument('-f',
                            help='Choose one or more groups of the following features: cs (Crowd sourcing), ibm (IBM), liwc (LIWC), tfidf (TFIDF). '
                                 'In case you choose more than one group, write the names with a space between them (default is cs).',
                            nargs='+', default='cs')
        parser.add_argument('-l',
                            help=f'Select the number of train/test splits (default={Commons.DEFAULT_NUM_LOOPS}).',
                            type=int, default=Commons.DEFAULT_NUM_LOOPS)
        parser.add_argument('-s',
                            help=f'Select the size of the train set in each split (default={Commons.DEFAULT_TRAIN_SIZE}).',
                            default=Commons.DEFAULT_TRAIN_SIZE)

        if len(sys.argv) == 1:
            parser.print_help(sys.stderr)
            sys.exit(1)
        args = parser.parse_args()
        game = args.g
        algorithm = args.a
        features = args.f
        num_loops = args.l
        train_size = args.s

        if game not in ['chicken', 'box', 'door']:
            raise Exception("The game name isn't correct")

        if algorithm not in ['tac', 'knn', 'svm']:
            raise Exception("The algorithm name is incorrect")

        for group in features:
            if group not in ['cs', 'ibm', 'liwc', 'tfidf']:
                raise Exception(f"{group} is not part of the features options")

        Run(game, algorithm, features, num_loops, train_size)

    except Exception as e:
        raise Exception("Exception occurred")
