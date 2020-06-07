import logging
import os
from string import punctuation

import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, accuracy_score

from commons import Commons


def evaluation(y_true, y_pred, labels_dict):
    """
    Evaluate a model using the measurements by Accuracy, F1 Macro and F1 Weight.
    :param y_true:
    :param y_pred:
    :param labels_dict: string names of the labels.
    :return: Dict with the measurements
    """
    unique_labels = list(set(y_true) | set(y_pred))
    accuracy = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro')
    f1_weight = f1_score(y_true, y_pred, average='weighted')
    f1_per_label = f1_score(y_true, y_pred, average=None, labels=unique_labels)
    result = {'Accuracy': accuracy,
              'F1 Macro': f1_macro,
              'F1 Weight': f1_weight,
              }

    for counter, key in enumerate(unique_labels):
        label = labels_dict[key]
        result['F1 ' + label] = f1_per_label[counter]

    # In case y_pred does not include one of the labels at all (for instance, predicting speed for all the observations
    # in Chicken), we set the F1 score per label manually. Using sk-learn's f1_score function in this case will result
    # in an exception.
    missing_labels = set(labels_dict.keys()) - set(unique_labels)
    for key in missing_labels:
        label = labels_dict[key]
        result['F1 ' + label] = np.NaN

    return result


def get_cluster_label_random_tie(x):
    """
    Get the majority label for each cluster. break ties randomly.
    :param x: dataframe with the observations and their labels in the clusters (after the split to clusters in TAC)
    :return: single label
    """
    # Get the count of each label in the cluster
    df_counts = x.value_counts().reset_index()
    # Save the count of the most frequent label in the cluster
    highest_count = df_counts.iloc[0, 1]
    # Get all the labels that appear in the same frequency in the cluster (the most frequent labels)
    df_tie = df_counts.loc[df_counts.iloc[:, 1] == highest_count]
    # For the most frequent labels in the cluster (with the highest frequency), break the tie and choose a label
    # randomly if there is only one label with a majority vote in the cluster, it will be returned
    label = np.random.choice(df_tie['index'].values)
    return label


def get_features(feature):
    if feature == 'cs':
        # Crowd sourcing features
        df = pd.read_csv(os.path.join(Commons.FILES_PATH, Commons.DATA_FOLDER_NAME, 'traits_features.csv'))
    elif feature == 'ibm':
        # IBM Personality Insights service features
        df = pd.read_csv(
            os.path.join(Commons.FILES_PATH, Commons.DATA_FOLDER_NAME, 'bluemix-tone-analyzer_features.csv'))
    elif feature == 'liwc':
        # LIWC features
        df = pd.read_csv(os.path.join(Commons.FILES_PATH, Commons.DATA_FOLDER_NAME, 'LIWC_features.csv'))
    elif feature == 'tfidf':
        # Calculate TFIDF on raw texts
        use_cols = ['orig_id', 'text']
        df_text = pd.read_csv(os.path.join(Commons.FILES_PATH, Commons.DATA_FOLDER_NAME, 'human_text_games.csv'),
                              usecols=use_cols)
        # Convert all letters to lower case
        df_text['filtered'] = df_text['text'].apply(lambda x: x.lower())
        # Remove punctuations marks
        table = str.maketrans(dict.fromkeys(punctuation))
        df_text['filtered'] = df_text['filtered'].apply(lambda x: x.translate(table))
        # Remove enters
        df_text['filtered'] = df_text['filtered'].apply(lambda x: x.replace('\r\r\n', ' '))
        df_text['filtered'] = df_text['filtered'].apply(lambda x: x.replace('\r\n\r\n', ' '))
        df_text['filtered'] = df_text['filtered'].apply(lambda x: x.replace('\n', ' '))
        # Remove stopwords
        df_text['filtered'] = df_text['filtered'].apply(
            lambda x: " ".join([word for word in x.split() if word not in stopwords.words('english')]))
        # Create TFIDF counter
        tf = TfidfVectorizer(smooth_idf=False, norm=None, min_df=3, max_df=0.9)
        document_list = df_text['filtered'].tolist()
        # Create tf-idf matrix - columns are documents and rows are words
        tfidf_matrix = tf.fit_transform(document_list)
        tfidf_df = pd.DataFrame(tfidf_matrix.toarray())
        df = pd.merge(df_text, tfidf_df, left_index=True, right_index=True, how='left')
        df.drop(['filtered', 'text'], axis=1, inplace=True)
        df.to_csv(os.path.join(Commons.FILES_PATH, Commons.DATA_FOLDER_NAME, 'tfidf_features.csv'), index=False)

    return df


def get_game_data(game, demographics=False):
    """
    Return the data of the chosen game.
    :param game: chicken, box, or door
    :param demographics: binary, whether to include gender and age in addition to the game data.
    :return: dataframe
    """
    # Upload the game data
    use_cols = ['orig_id', game]
    if demographics:
        use_cols.extend(['gender', 'age'])
    return pd.read_csv(os.path.join(Commons.FILES_PATH, Commons.DATA_FOLDER_NAME, 'human_text_games.csv'),
                       usecols=use_cols)


def get_trait_df(features):
    f = []
    df_traits = None
    features = [features] if isinstance(features, str) else features
    for feature in features:
        f.append(get_features(feature))
    if len(features) == 1:
        df_traits = f[0]
    elif len(features) > 1:
        df_traits = pd.merge(*f, on='orig_id')
    if df_traits is None:
        raise Exception("get_trait_df: Could not create the traits DataFrame")
    return df_traits


def preprocess_traits(df_traits):
    """
    Scaling features columns between 0 to 1
    """
    # Drop unnecessary column name
    traits = df_traits.columns.drop(['orig_id']).tolist()
    # Create a minimum and maximum processor object
    min_max_scaler = preprocessing.MinMaxScaler()
    # Create an object to transform the data to fit the scaler processor
    x_scaled = min_max_scaler.fit_transform(df_traits[traits])
    # Run the scaler on the dataframe
    df_scaled = pd.DataFrame(x_scaled, columns=traits)
    # Assign back to the original df
    df_traits[df_scaled.columns] = df_scaled
    return df_traits


def verify_folders():
    try:
        if not os.path.exists(Commons.RESULTS_PATH):
            os.makedirs(Commons.RESULTS_PATH)
        if not os.path.exists(Commons.TRANS_SVM_PATH):
            os.makedirs(Commons.TRANS_SVM_PATH)
    except Exception as e:
        print("Exception in utils.verify_folders: " + str(e))


def get_logger():
    # Set logger configurations
    logger = logging.getLogger(Commons.LOGGER_NAME)
    logger.setLevel(logging.INFO)
    # create a file handler
    handler = logging.FileHandler(Commons.LOG_FILENAME)
    handler.setLevel(logging.INFO)
    # create a logging format
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    # add the file handler to the logger
    logger.addHandler(handler)
    return logger
