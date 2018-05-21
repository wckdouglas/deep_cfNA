#!/usr/bin/env python

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
import random
from deep_cfNA.deep_cfNA_model import deep_cfNA, load_model, save_model, plot_train


def main():
    work_dir = '/stor/work/Lambowitz/cdw2854/cell_Free_nucleotides/tgirt_map/classifier'
    test_bed = work_dir + '/test.bed'
    train_bed = work_dir + '/train.bed'
    fa_file = '/stor/work/Lambowitz/ref/hg19/genome/hg19_genome.fa'
    model_prefix = work_dir + '/deef_cfNA'

    train = True

    if train:
        history, model = training_sample(train_bed, fa_file)
        save_model(model, prefix = model_prefix)
        plot_train(history)

    else:
        model = load_model(model_prefix)

    validation_sample(test_bed, fa_file, model)



if __name__ == '__main__':
    main()
