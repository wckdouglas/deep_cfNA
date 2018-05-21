from keras.models import Sequential, model_from_json
from keras.layers import Dense, Conv1D,\
                         Flatten, MaxPool1D,  \
                         Dropout, LSTM, \
                         Bidirectional
from deep_cfNA.metrics import f1_metrics
from deep_cfNA.bed_utils import frag_size
import sys

def deep_cfNA():
    '''
    DanQ model 
    https://github.com/uci-cbcl/DanQ/blob/master/DanQ-JASPAR_train.py
    '''
    model = Sequential()
    model.add(Conv1D(filters=160, 
                  kernel_size = 26,
                  strides = 1,
                  padding = 'valid',
                  input_shape = (frag_size,5),
                  activation = 'relu'))
    model.add(MaxPool1D(pool_size=50, strides=13))
    model.add(Dropout(0.2))
    model.add(Bidirectional(LSTM(64)))
    model.add(Dropout(0.5))
    model.add(Dense(25, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', 
                optimizer='rmsprop', 
                metrics=[f1_metrics,'binary_accuracy'])
    return model

def load_model(prefix):
    '''
    keras load model
    '''
    json = open(prefix + '_architecture.json', 'r').read()
    model = model_from_json(json)
    model.load_weights(prefix + '_weights.h5')
    print('Load model: %s' %prefix, file = sys.stderr)
    return model


def save_model(model, prefix = 'model'):
    '''
    keras save model, weight
    '''
    weight_h5 = prefix + '_weights.h5'
    model.save_weights(weight_h5)
    print('Saved weights to %s' %weight_h5, file = sys.stderr)

    # Save the model architecture
    model_json = prefix + '_architecture.json'
    with open(model_json, 'w') as f:
        f.write(model.to_json())

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
