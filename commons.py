import os


class Commons:
    FILES_PATH = os.path.dirname(os.path.realpath(__file__))
    DATA_FOLDER_NAME = 'data'
    RESULTS_PATH = os.path.join(FILES_PATH, 'results')
    TRANS_SVM_PATH = os.path.join(FILES_PATH, 'transductive_svm')
    DEFAULT_NUM_LOOPS = 5000
    DEFAULT_TRAIN_SIZE = 0.9
    LOG_FILENAME = 'Predicting-NLPGT.log'
    LOGGER_NAME = 'Predicting-NLPGT-Logger'
