from deep_cfNA.deep_cfNA_model import deep_cfNA, load_model, save_model, plot_train
from deep_cfNA.bed_utils import generate_padded_data, data_generator
from collections import defaultdict
import numpy as np

def training_sample(train_bed, fa_file):
    '''
    Set up keras model
    retrieve data and train
    '''
    
    model = deep_cfNA()
    history = model.fit_generator(data_generator(train_bed, 
                                                 fa_file, 
                                                batch_size = 500),
                                  epochs = 20,
                                  steps_per_epoch = 1000)


    print('Fitted model')
    return history, model

def fetch_validation(test_bed, fa_file, batch_size = 0):
    '''
    fetch sequences from test bed file and return feature arrays and test label
    '''
    label_count = defaultdict(int)
    if batch_size > 0:
        features = []
        labels = []
        for seq, label in generate_padded_data(test_bed, fa_file):
            if label_count[label] < batch_size/2:
                features.append(seq)
                labels.append(label)
                label_count[label] += 1
    
    else:
        data = [data for data in generate_padded_data(test_bed, fa_file)]
        features, labels = zip(*data)

    features = np.array(features)
    labels = np.array(labels)
    return features, labels


def validation_sample(test_bed, fa_file, model):
    '''
    Test model on unseen data
    '''
    # validation of the model
    X_test, y_test = fetch_validation(test_bed, fa_file)
    print('Fetched test samples')

    # make prediction
    y_pred_prob = model.predict_proba(X_test)
    y_pred_prob = y_pred_prob.flatten()

    y_pred_class = model.predict_classes(X_test)
    y_pred_class = y_pred_class.flatten()

    # evaluation
    print("[Validation] Precision: %1.3f" % precision_score(y_test, y_pred_class))
    print("[Validation] Recall: %1.3f" % recall_score(y_test, y_pred_class))
    print("[Validation] F1: %1.3f" % f1_score(y_test, y_pred_class))
    print("[Validation] AUROC: %1.3f" % roc_auc_score(y_test, y_pred_prob))