from keras.models import Sequential, model_from_json
from keras.layers import Dense, Conv1D,\
                         Flatten, MaxPool1D,  \
                         Dropout, LSTM, \
                         Bidirectional
from .metrics import f1_score, precision, recall
from .bed_utils import frag_size
import sys

class Deep_cfNA():
    '''
    DanQ model 
    https://github.com/uci-cbcl/DanQ/blob/master/DanQ-JASPAR_train.py
    '''

    def __init__(self, reduced_model=False):
        self.Model = Sequential()
        
        if not reduced_model:
            self.Model.add(Conv1D(filters=160, 
                          kernel_size = 26,
                          strides = 1,
                          padding = 'valid',
                          input_shape = (frag_size,5),
                          use_bias=False,
                          activation = 'relu')) # no padding: output Tensor: (-1, 375, 160)
            self.Model.add(MaxPool1D(pool_size=40, strides=12)) #pooled: (-1, ,160)
            self.Model.add(Dropout(0.2)) #only happens during training
            self.Model.add(Bidirectional(LSTM(64, return_sequences=True))) #output (-1, 128)
            self.Model.add(Flatten())
            self.Model.add(Dropout(0.5)) 
            self.Model.add(Dense(50, activation='relu')) 
            self.Model.add(Dense(25, activation='relu'))
            self.Model.add(Dropout(0.2)) 
            self.Model.add(Dense(10, activation='relu'))
            self.Model.add(Dense(1, activation='sigmoid'))

        else:
            self.Model.add(Conv1D(filters=160, 
                          kernel_size = 26,
                          strides = 1,
                          padding = 'valid',
                          input_shape = (frag_size,5),
                          use_bias=False,
                          activation = 'relu')) # no padding: output Tensor: (-1, 375, 160)
            self.Model.add(MaxPool1D(pool_size=40, strides=12)) #pooled: (-1, ,160)
            self.Model.add(Dropout(0.2)) #only happens during training
            self.Model.add(Flatten())
            self.Model.add(Dense(50, activation='relu'))
            self.Model.add(Dropout(0.2)) 
            self.Model.add(Dense(1, activation='sigmoid'))
    
    
    def compile(self):
        '''
        compile keras model
        '''
        self.Model.compile(loss='binary_crossentropy', 
                    optimizer='Adam', 
                    metrics=[f1_score, 
                             precision, 
                             recall,
                             'binary_accuracy'])


    def predict_classes(self, *args, **kwargs):
        return self.Model.predict_classes(*args, **kwargs)

    def predict(self, *args, **kwargs):
        return self.Model.predict_proba(*args, **kwargs)


    def predict_proba(self, *args, **kwargs):
        return self.Model.predict_proba(*args, **kwargs)

    def fit_generator(self,*args, **kwargs):
        return self.Model.fit_generator(*args, **kwargs)

    def load_model(self, prefix, message=True):
        '''
        keras load model
        '''
        json = open(prefix + '_architecture.json', 'r').read()
        self.Model = model_from_json(json)
        self.Model.load_weights(prefix + '_weights.h5')
        if message:
            print('Loaded model: %s' %prefix, file = sys.stderr)


    def save_model(self, prefix = 'model'):
        '''
        keras save model, weight
        '''
        weight_h5 = prefix + '_weights.h5'
        self.Model.save_weights(weight_h5)
        print('Saved weights to %s' %weight_h5, file = sys.stderr)

        # Save the model architecture
        model_json = prefix + '_architecture.json'
        with open(model_json, 'w') as f:
            f.write(self.Model.to_json())

        print('Saved model to %s' %model_json, file = sys.stderr)



def plot_train(history):
    from matplotlib import use as mpl_use
    mpl_use('Agg')
    import seaborn as sns
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(111)
    for key, vals in history.history.items():
        ax.plot(np.arange(len(vals)), vals, '-o',  label = key)
    ax.legend()
    fig.savefig('deep_train.png', bbox_inches='tight', transparent = True)
    return 0
