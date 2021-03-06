from .model import Deep_cfNA
from .bed_utils import generate_padded_data, data_generator, random
from collections import defaultdict
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from keras.callbacks import TensorBoard, EarlyStopping


def training_sample(train_bed_pos, train_bed_neg, fa_file, 
                    steps = 10000,
                    epochs = 5, 
                    model = Deep_cfNA(),
                    N_padded=True, validation_bed = None):
    '''
    Set up keras model
    retrieve data and train
    '''
    
    tensorboard = TensorBoard(log_dir='./tensorboard', histogram_freq=0,
                              write_graph=True, write_images=False)
    callbacks = [tensorboard]

    if validation_bed:
        '''
        get all records from validation
        '''
        X_val, Y_val = [], []
        for x, y in fetch_validation(validation_bed, fa_file):
            X_val.extend(x)
            Y_val.extend(y)
        
        X_val, Y_val = np.array(X_val), np.array(Y_val)
        Y_val = Y_val.reshape(-1,1)
        validation_data = (X_val, Y_val)

        early_stop = EarlyStopping(monitor='loss') #val loss doesnt help
        print('[Using early stop] Fetched n=%i validation data' %(len(Y_val)))

    else:
        validation_data = None
        early_stop = EarlyStopping(monitor='loss')
    callbacks.append(early_stop)


    #get train data generator
    train_data = data_generator(train_bed_pos, 
                             train_bed_neg,
                             fa_file, 
                             batch_size = 500,
                             N_padded = N_padded)

    #Training here
    history = model.fit_generator(train_data,
                                  epochs = epochs,
                                  steps_per_epoch = steps,
                                  validation_data = validation_data,
                                  callbacks = callbacks)
    y_pred = model.predict(X_val)
    print(Y_val.shape, y_pred.shape)


    print('Fitted model')
    return history, model


def feature_generator(test_bed, fa_file, batch_size):
    features = []
    labels = []
    for sample_number, (seq, label) in enumerate(generate_padded_data(test_bed, fa_file)):
        features.append(seq)
        labels.append(label)
        if sample_number % batch_size == 0 and sample_number > 0:
            yield np.array(features), np.array(labels)
            features = []
            labels = []
    
    yield np.array(features), np.array(labels)
    


def fetch_validation(test_bed, fa_file, batch_size = 1000):
    '''
    fetch sequences from test bed file and return feature arrays and test label
    if batch_size <= 0, then return full data
    '''
    features = []
    labels = []

    if batch_size > 0:
        # return generator
        return(feature_generator(test_bed, fa_file, batch_size))

    else:
        # return full data
        print('Using full data')
        data = [data for data in generate_padded_data(test_bed, fa_file)]
        features, labels = list(zip(*data))

        features = np.array(features)
        labels = np.array(labels)
        return features, labels


def validation_sample(test_bed, fa_file, model):
    '''
    Test model on unseen data
    '''
    # validation of the model
    true_label = []
    predicted_prob = []
    samples = 0

    for X_test, y_test in fetch_validation(test_bed, fa_file):

        # make prediction
        y_pred_prob = model.predict_proba(X_test)
        y_pred_prob = y_pred_prob.flatten()
        assert(len(y_pred_prob) == len(y_test))
        predicted_prob.extend(y_pred_prob.tolist())
        true_label.extend(y_test.tolist())
        samples += len(y_test)

    predicted_prob = 1 - np.array(predicted_prob)
    true_label = 1 - np.array(true_label)
    assert(predicted_prob.shape == true_label.shape)

    predicted_class = predicted_prob.round()

    # evaluation
    print('[Validation] On %i samples:' %samples)
    print("[Validation] Precision: %1.3f" % precision_score(true_label, predicted_class))
    print("[Validation] Recall: %1.3f" % recall_score(true_label, predicted_class))
    print("[Validation] F1: %1.3f" % f1_score(true_label, predicted_class))
    print("[Validation] AUROC: %1.3f" % roc_auc_score(true_label, predicted_prob))



   
