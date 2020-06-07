"""
Replicating the results of the paper
"""
import os

import matplotlib.pyplot as plt
import pandas as pd

from commons import Commons
from main import Run


def Figure5():
    metrics = ['F1 Macro', 'F1 Weight', 'Accuracy']
    for game in ['chicken', 'box', 'door']:
        algo_results = []
        for algorithm in ['tac', 'mvc', 'erg', 'ewg']:
            if algorithm == 'tac':
                tac_result = Run(game, algorithm, 'cs', Commons.DEFAULT_NUM_LOOPS, Commons.DEFAULT_TRAIN_SIZE)
            else:
                result = Run(game, algorithm)
                result['algorithm'] = algorithm
                algo_results.append(result[metrics + ['algorithm']])
        algo_results = pd.concat(algo_results, axis=0)

        fig = plt.figure(figsize=([25, 6]))
        for i, metric in enumerate(metrics, 1):
            ax = fig.add_subplot(1, 3, i)
            tac_result.plot(x='NumClusters', y=[metric], style='^-', color='r', title=game, ax=ax)
            ax.axhline(y=algo_results[algo_results.algorithm == 'mvc'][metric][0], color='b', linestyle='--')
            ax.axhline(y=algo_results[algo_results.algorithm == 'erg'][metric][0], color='g', linestyle='--')
            ax.axhline(y=algo_results[algo_results.algorithm == 'ewg'][metric][0], color='brown', linestyle='--')
            _, labels = ax.get_legend_handles_labels()
            labels[:2] = ['TAC', 'MVC', 'ERG', 'EWG']
            ax.legend(labels=labels)
            ax.set_xlabel('Number of Clusters')
            ax.set_ylabel(metric)
        plt.savefig(os.path.join(Commons.RESULTS_PATH, 'Figure5_' + game + '.png'))
        plt.show()


def Table2():
    numClustDict = {'chicken': 13, 'box': 30, 'door': 26}
    final_results = []
    for algorithm in ['tac', 'mvc', 'erg', 'ewg']:
        games_results = []
        for game in ['chicken', 'box', 'door']:
            if algorithm == 'tac':
                result = Run(game, algorithm, 'cs', Commons.DEFAULT_NUM_LOOPS, Commons.DEFAULT_TRAIN_SIZE)
                result = result[result.NumClusters == numClustDict[game]].reset_index(drop=True)
                result.drop('NumClusters', axis=1, inplace=True)
            else:
                result = Run(game, algorithm)
            games_results.append(result.drop(['Accuracy', 'F1 Macro', 'F1 Weight'], axis=1))

        games_results = pd.concat(games_results, axis=1)
        games_results['algorithm'] = algorithm
        final_results.append(games_results)

    final_results = pd.concat(final_results, axis=0)
    final_results.to_csv(os.path.join(Commons.RESULTS_PATH, 'Table2.csv'), index=False)


def Table3():
    args = [
        ['tac', 'cs', [13, 30, 26], [6, 13, 17]],
        ['knn', 'cs', [1, 1, 1], [2, 5, 5]],
        ['svm', 'cs', [], []],
        ['tac', 'ibm', [30, 25, 8], [11, 9, 18]],
        ['tac', ['cs', 'ibm'], [4, 23, 19], [12, 16, 9]],
        ['tac', 'liwc', [17, 11, 14], [30, 15, 24]],
        ['tac', ['cs', 'liwc'], [28, 20, 30], [9, 16, 13]],
        ['tac', 'tfidf', [28, 25, 28], [11, 12, 11]]
    ]

    games = ['chicken', 'box', 'door']
    metrics = ['Accuracy', 'F1 Macro', 'F1 Weight']
    final_upper_results = []
    final_lower_results = []
    for algorithm, features, paramsUpper, paramsLower in args:
        for i, game in enumerate(games):
            result = Run(game, algorithm, features, Commons.DEFAULT_NUM_LOOPS, Commons.DEFAULT_TRAIN_SIZE)
            if algorithm == 'svm':
                result = result.to_frame().T
            result['game'] = game
            result['algorithm'] = algorithm + ' ' + str(features)
            if algorithm == 'tac':
                upper_result = result[result.NumClusters == paramsUpper[i]].reset_index(drop=True)
                lower_result = result[result.NumClusters == paramsLower[i]].reset_index(drop=True)
            elif algorithm == 'knn':
                upper_result = result[result.neighbors == paramsUpper[i]].reset_index(drop=True)
                lower_result = result[result.neighbors == paramsLower[i]].reset_index(drop=True)
            else:
                upper_result = result
                lower_result = result
            final_upper_results.append(upper_result[metrics + ['game', 'algorithm']])
            final_lower_results.append(lower_result[metrics + ['game', 'algorithm']])

    for i, df in enumerate([final_upper_results, final_lower_results]):
        df = pd.concat(df, axis=0)
        df = df.pivot(index='algorithm', columns='game', values=metrics)
        df.to_csv(os.path.join(Commons.RESULTS_PATH, 'Table3_' + str(i) + '.csv'))
