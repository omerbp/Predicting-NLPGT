from collections import Counter

import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.model_selection import train_test_split

from utils import get_cluster_label_random_tie, evaluation


# Run TAC(Transductive Attribute Clustering)
def transductive_clustering(x, y, game, labels_dict, num_loops, train_size):
    """
    Run TAC(Transductive Attribute Clustering)
    :param x: features
    :param y: label
    :param game: game name
    :param labels_dict: dictionary with keys as integers and values as labels
    :param num_loops: number of train/test splits
    :param train_size: size of the train set in each split
    :return: dataframe with the average evaluation measures across all loops
    """
    loop_results = pd.DataFrame()
    for num_clusters in range(2, 31):
        hc = AgglomerativeClustering(n_clusters=num_clusters, affinity='euclidean', linkage='ward')
        y_hc = hc.fit_predict(x)
        x['cluster'] = y_hc
        for n in range(num_loops):
            x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                                train_size=train_size,
                                                                test_size=1. - train_size,
                                                                shuffle=True)
            # Get label for each cluster
            df_freq = pd.concat([x_train['cluster'], y_train.rename(game)], axis=1).groupby('cluster')[game].apply(
                lambda x: get_cluster_label_random_tie(x)).reset_index()
            # Save a dictionary where the key is a cluster and the value is the chosen label for the cluster
            dict_freq = dict(zip(df_freq.cluster, df_freq[game]))
            # Save the most frequent label across all clusters
            freq_label, count = Counter(dict_freq.values()).most_common(1)[0]
            # In case there is an observation in the test set with a cluster that don't appear in the train set,
            # the predicted label for her is the most frequent label across all clusters
            cluster_predict = x_test.cluster.apply(
                lambda x: dict_freq[x] if x in dict_freq else freq_label)
            x_test = x_test.assign(cluster_predict=cluster_predict)
            eval_dict = evaluation(y_test,
                                   x_test['cluster_predict'].values,
                                   labels_dict)
            eval_dict['NumClusters'] = num_clusters
            loop_results = loop_results.append(pd.DataFrame.from_dict(eval_dict, orient='index').T)

    # Calculate the average evaluation metrics across all loops
    loop_results = loop_results.groupby('NumClusters').mean().reset_index().apply(pd.to_numeric)
    return loop_results
