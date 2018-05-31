from deep_cfNA.model import Deep_cfNA
from deep_cfNA.bed_utils import generate_padded_data, data_generator, random
from collections import defaultdict
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from keras.callbacks import TensorBoard


def training_sample(train_bed_pos, train_bed_neg, fa_file, 
                    steps = 10000,
                    epochs = 5, 
                    N_padded=True, validation_bed = None):
    '''
    Set up keras model
    retrieve data and train
    '''
    
    tensorboard = TensorBoard(log_dir='./tensorboard', histogram_freq=0,
                              write_graph=True, write_images=False)
    model = Deep_cfNA()

    if validation_bed:
        validation_generator = fetch_validation(validation_bed, fa_file)
        validation_step = 50
    else:
        validation_generator = None
        validation_step = None

    history = model.fit_generator(data_generator(train_bed_pos, 
                                                 train_bed_neg,
                                                 fa_file, 
                                                batch_size = 500,
                                                N_padded = N_padded),
                                  epochs = epochs,
                                  steps_per_epoch = steps,
                                  validation_data = validation_generator,
                                  validation_steps = validation_step,
                                  callbacks = [tensorboard])


    print('Fitted model')
    return history, model

def fetch_validation(test_bed, fa_file, batch_size = 1000):
    '''
    fetch sequences from test bed file and return feature arrays and test label
    '''
    features = []
    labels = []
    for sample_number, (seq, label) in enumerate(generate_padded_data(test_bed, fa_file)):
        if random() > 0.5:
            features.append(seq)
            labels.append(label)
            if sample_number % batch_size == 0 and sample_number > 0:
                yield np.array(features), np.array(labels)
                features = []
                labels = []
    
    yield np.array(features), np.array(labels)


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

    predicted_prob = np.array(predicted_prob)
    true_label = np.array(true_label)
    assert(predicted_prob.shape == true_label.shape)

    predicted_class = predicted_prob.round()

    # evaluation
    print('[Validation] On %i samples:' %samples)
    print("[Validation] Precision: %1.3f" % precision_score(true_label, predicted_class))
    print("[Validation] Recall: %1.3f" % recall_score(true_label, predicted_class))
    print("[Validation] F1: %1.3f" % f1_score(true_label, predicted_class))
    print("[Validation] AUROC: %1.3f" % roc_auc_score(true_label, predicted_prob))



   
