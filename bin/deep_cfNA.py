#!/usr/bin/env python

import numpy as np
import random
from deep_cfNA import deep_cfNA_training
import argparse
import pkg_resources
from deep_cfNA.deep_cfNA_model import save_model
from deep_cfNA.deep_cfNA_training import training_sample, validation_sample
from deep_cfNA.bed_utils import prediction_generator

def getopt():
    parser = argparse.ArgumentParser(prog='Deep cfNA model')
    subparsers = parser.add_subparsers(help='Run type for deep cfNA',
                                    dest='subcommand')
    subparsers.required = True

    #add training commands
    training = subparsers.add_parser('train',help='deep_cfNA training')
    training.add_argument('--train_bed', help='bed file for fragments:  \n'\
                        'For each record in bed file, extract the sequence, and center it \n'\
                        'fill up both sides to length of (seq_length) with Ns.\n' \
                        'Bed files need these columns: \n'\
                        '1. chrom \n'\
                        '2. start\n'\
                        '3. end\n'\
                        '4.\n'\
                        '5. \n'\
                        '6. strand\n'\
                        '7. label: (DNA or RNA)',
                        required=True)
    training.add_argument('--genome', help ='genome fasta file (must have faidx)', required=True)
    training.add_argument('--validation_bed', help ='bed file for validation')
    training.add_argument('--model_prefix', help ='where to save the model')


    #add_test commands
    prediction = subparsers.add_parser('predict', help = 'deep cfNA prediction using existing model')
    prediction.add_argument('--inbed',help='bed file for fragments:  \n'\
                        'For each record in bed file, extract the sequence, and center it \n'\
                        'fill up both sides to length of (seq_length) with Ns.\n' \
                        'Bed files need these columns: \n'\
                        '1. chrom \n'\
                        '2. start\n'\
                        '3. end\n'\
                        '4.\n'\
                        '5. \n'\
                        '6. strand\n'\
                        '7. label: (DNA or RNA)',
                        required=True)
    prediction.add_argument('--genome', help ='genome fasta file (must have faidx)', required=True)
    prediction.add_argument('--model_prefix', default=pkg_resources.resource_filename('deep_cfNA', "model/deep_cfNA"))

    return parser.parse_args()
        

def main():
    args = getopt()

    if args.subcommand == 'train':
        history, model = training_sample(args.train_bed, args.genome)
        save_model(model, prefix = args.model_prefix)
        validation_sample(args.validation_bed, args.genome, model)


    elif args.subcommand == 'predict':
        model = load_model(args.model_prefix)
        bed_generator = prediction_generator(args.inbed, args.genome, batch_size = 1000)
        predictions = model.prediction_generator(bed_generator)
        with open(args.inbed, 'r') as bed:
            for line, prediction in zip(bed, predicitons):
                na = 'DNA' if prediction == 1 else 'RNA'
                outline = line.strip() +'\t' + na
                print(outline)
        





if __name__ == '__main__':
    main()
