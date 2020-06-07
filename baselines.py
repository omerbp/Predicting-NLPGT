import itertools
import os
import time
from math import ceil
from subprocess import call

import numpy as np
import pandas as pd
from sklearn.datasets import dump_svmlight_file
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from commons import Commons
from utils import evaluation


def _baselines_evaluation(df_conf_mat, y, labels, labels_dict):
    df_conf_mat.set_index('class', inplace=True)
    df_conf_mat.drop(['game'], axis=1, inplace=True)
    recall = np.diag(df_conf_mat) / df_conf_mat.sum(axis=1).values
    precision = np.diag(df_conf_mat) / df_conf_mat.sum().values
    dict_f1 = {}
    for label in labels:
        dict_f1['F1 ' + labels_dict[label]] = (2 * recall[label] * precision[label]) / (
                    recall[label] + precision[label])
    f1_macro = np.nansum(list(dict_f1.values())) / len(labels)
    weights = (y.value_counts().to_frame() / len(y)).sort_index().values
    f1_weight = np.nansum(np.multiply(weights, np.array(list(dict_f1.values())).reshape(len(labels), 1)))
    accuracy = np.sum(np.diag(df_conf_mat)) / len(y)
    dict_f1.update({'Accuracy': accuracy,
                    'F1 Macro': f1_macro,
                    'F1 Weight': f1_weight})

    return dict_f1


def baselines(y, algorithm, labels_dict):
    df_count = y.value_counts().reset_index().rename(columns={'index': 'class', 0: 'game'})
    df_count.sort_values('class', inplace=True)
    labels = df_count['class'].tolist()
    # create confusion matrix
    for label in labels[:-1]:
        if algorithm == 'mvc':
            majority_idx = y.value_counts().index[0]
            weights = np.zeros(len(labels))
            weights[majority_idx] = 1
            df_count[label] = df_count['game'].copy() * weights[label]
        elif algorithm == 'erg':
            df_count[label] = df_count['game'].copy() / len(labels)
        elif algorithm == 'ewg':
            weights = (y.value_counts().to_frame() / len(y)).sort_index().values
            df_count[label] = df_count['game'].copy() * weights[label]
        df_count[label] = df_count[label].apply(lambda x: ceil(x))

    df_count[labels[-1]] = df_count['game'] - df_count[labels[:-1]].sum(axis=1)
    df_result = pd.DataFrame.from_dict(_baselines_evaluation(df_count, y, labels, labels_dict), orient='index').T
    df_result.fillna(0, inplace=True)

    return df_result


def _get_label(row):
    """
    Auxiliary function required for multi-class transductive SVM
    :param row: single observation from the test set that contains results from 3 runs of door game.
                Assign label according to majority vote, else break tie by random choice.
    :return: single label
    """
    # If there is no tie for any of the observations we get only one column
    if len(row) == 1:
        return row[0]
    # If the second run (column 1) is null then there is only one majority label (column 0)
    elif pd.isnull(row[1]):
        return row[0]
    else:
        # Choose random column number and return it's label
        label = np.random.choice(range(1, len(row)))
        return row[label]


def knn_classifier(x, y, game, labels_dict, num_loops, train_size):
    """
    Runs KNN classifier on the data
    :param x: features
    :param y: label
    :param game: game name
    :param labels_dict: dictionary with keys as integers and values as labels
    :param num_loops: number of train/test splits
    :param train_size: size of the train set in each split
    :return: dataframe with the average evaluation measures across all loops
    """
    loop_results = pd.DataFrame()
    for k in range(1, 6):
        for n in range(num_loops):
            # Split train test
            x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                                train_size=train_size,
                                                                test_size=1. - train_size,
                                                                shuffle=True)
            # Create and fit a nearest-neighbor classifier - k neighbors
            knn = KNeighborsClassifier(k)
            knn.fit(x_train, y_train)
            prediction = knn.predict(x_test)
            # Compute evaluation scores
            eval_dict = evaluation(y_test, prediction, labels_dict)
            eval_dict['neighbors'] = k
            loop_results = loop_results.append(pd.DataFrame.from_dict(eval_dict, orient='index').T)

    # Calculate the average evaluation metrics across all loops
    loop_results = loop_results.groupby('neighbors').mean().reset_index()
    loop_results.to_csv(os.path.join(Commons.RESULTS_PATH, 'KNN_' + game + '.csv'), index=False)
    return loop_results


def transductive_svm(x, y, game, num_loops, train_size):
    """
    This function calls SVMlight by Thorsten Joachims. More information can be found here: http://svmlight.joachims.org/
    The files should be placed under a folder named transductive_svm.
    :param x: features
    :param y: label
    :param num_loops: number of train/test splits
    :param train_size: size of the train set in each split
    :return: dataframe with the average evaluation measures across all loops
    """
    if game == 'chicken':
        y = y.apply(lambda x: -1 if x == 'Speed!' else 1)
        labels_dict = {-1: 'Speed!', 1: 'Stop!'}
    else:
        y = y.apply(lambda x: -1 if x == 'The Left box' else 1)
        labels_dict = {-1: 'The Left box', 1: 'The Right box'}

    # Define path to files
    train_file = os.path.join(Commons.TRANS_SVM_PATH, 'train.txt')
    model_file = os.path.join(Commons.TRANS_SVM_PATH, 'model.txt')
    labels_file = os.path.join(Commons.TRANS_SVM_PATH, 'labels.txt')
    svm_learn = os.path.join(Commons.TRANS_SVM_PATH, 'svm_learn')

    y = y.to_frame()
    loop_results = pd.DataFrame()
    for n in range(num_loops):
        # Split to train and test sets
        x_train, _, _, y_test = train_test_split(x, y[game].copy(),
                                                 train_size=train_size,
                                                 test_size=1. - train_size,
                                                 shuffle=True)

        ids_labeled = x_train.index.tolist()
        # Test observations get label=0
        y['new_label'] = y.apply(lambda row: row[game] if row.name in ids_labeled else 0, axis=1)
        # Write files in SVMlight format
        dump_svmlight_file(x, y['new_label'], train_file, zero_based=False)
        if not os.path.exists(svm_learn):
            raise Exception(
                "transductive_svm: svm_learn script doesn't exists. Need to download SVMlight by Thorsten Joachims "
                "and place the files under a folder named transductive_svm. More information can be found here: "
                "http://svmlight.joachims.org/ ")
        call(svm_learn + " -l " + labels_file + " " + train_file + " " + model_file, shell=True)
        # Need to wait until the file of prediction is created in order to do evaluation.
        while not os.path.isfile(labels_file):
            time.sleep(1)

        df_pred = pd.read_csv(labels_file, header=None, sep=' ')
        df_pred['prediction'] = df_pred[0].apply(lambda x: int(x[-2:]))
        predictions = df_pred['prediction'].values
        # Keep same order as in the training file
        y_test = y_test.sort_index()
        eval_dict = evaluation(y_test, predictions, labels_dict)
        loop_results = loop_results.append(pd.DataFrame.from_dict(eval_dict, orient='index').T)
        if os.path.isfile(labels_file):
            os.remove(labels_file)

    # Calculate the average evaluation metrics across all loops and save results to file
    loop_results = loop_results.mean()
    loop_results.to_csv(os.path.join(Commons.RESULTS_PATH, 'TransSVM_' + game + '.csv'), index=False)
    return loop_results


def transductive_svm_multiclass(x, y, game, num_loops, train_size):
    """
    This function calls SVMlight by Thorsten Joachims. More information can be found here: http://svmlight.joachims.org/
    The files should be placed under a folder named transductive_svm.
    :param x: features
    :param y: label
    :param num_loops: number of train/test splits
    :param train_size: size of the train set in each split
    :return: dataframe with the average evaluation measures across all loops
    """
    # Define path to files
    train_file = os.path.join(Commons.TRANS_SVM_PATH, 'train.txt')
    model_file = os.path.join(Commons.TRANS_SVM_PATH, 'model.txt')
    labels_file = os.path.join(Commons.TRANS_SVM_PATH, 'labels.txt')
    svm_learn = os.path.join(Commons.TRANS_SVM_PATH, 'svm_learn')

    y = y.to_frame()
    loop_results = pd.DataFrame()
    for n in range(num_loops):
        # Split to train and test sets
        x_train, _, _, y_test = train_test_split(x, y[game].copy(),
                                                 train_size=train_size,
                                                 test_size=1. - train_size,
                                                 shuffle=True)
        ids_labeled = x_train.index.tolist()
        # Test observations get label=0
        y['new_label'] = y.apply(lambda row: row[game] if row.name in ids_labeled else 0, axis=1)
        # Original test set to predict
        df_all_predictions = y[y['new_label'] == 0][[game]]
        for counter, group in enumerate(itertools.combinations(y[game].unique(), 2)):
            # Run SVMlight with 2 classes in each iteration. Get the 2 classes from the train set + test set
            new_idx = y[(y['new_label'].isin(group)) | (y['new_label'] == 0)].index
            df_new = pd.concat([x, y], axis=1)
            df_new = df_new[df_new.index.isin(new_idx)]
            labels_dict = dict(zip(group, (1, -1)))
            df_new['new_label'] = df_new['new_label'].apply(lambda x: labels_dict[x] if x in labels_dict.keys() else x)

            if not os.path.exists(svm_learn):
                raise Exception(
                    "transductive_svm: svm_learn script doesn't exists. Need to download SVMlight by Thorsten Joachims "
                    "and place the files under a folder named transductive_svm. More information can be found here: "
                    "http://svmlight.joachims.org/ ")
            call(svm_learn + " -l " + labels_file + " " + train_file + " " + model_file, shell=True)
            # Need to wait until the file of prediction is created in order to do evaluation.
            while not os.path.isfile(labels_file):
                time.sleep(1)

            # Read rhe predictions from the file
            df_pred = pd.read_csv(labels_file, header=None, sep=' ')
            df_pred['prediction'] = df_pred[0].apply(lambda x: int(x[-2:]))
            invert_labels_dict = {v: k for k, v in labels_dict.items()}
            df_pred['prediction'] = df_pred['prediction'].apply(lambda x: invert_labels_dict[x])
            df_all_predictions['pred' + str(counter)] = df_pred['prediction'].values
            if os.path.isfile(labels_file):
                os.remove(labels_file)

        # Get majority vote label, in case there is a tie --> the tie breaking rule is random choice
        votes = df_all_predictions.iloc[:, 1:].mode(axis=1)
        votes['pred'] = votes.apply(lambda x: _get_label(x), axis=1)
        predictions = votes['pred'].values
        true_labels = df_all_predictions[game].values
        labels = list(set(true_labels) | set(predictions))
        new_labels_dict = dict(zip(labels, labels))
        eval_dict = evaluation(true_labels, predictions, new_labels_dict)
        loop_results = loop_results.append(pd.DataFrame.from_dict(eval_dict, orient='index').T)

    # Calculate the average evaluation metrics across all loops and save results to file
    loop_results = loop_results.mean()
    loop_results.to_csv(os.path.join(Commons.RESULTS_PATH, 'TransSVM_' + game + '.csv'))
    return loop_results
