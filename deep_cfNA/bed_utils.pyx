from __future__ import print_function
from operator import itemgetter
from collections import defaultdict
from sequencing_tools.fastq_tools import reverse_complement, \
                    onehot_sequence_encoder
import pysam
import numpy as np
import sys
import random
from sequencing_tools.io_tools import xopen
from libc.stdint cimport uint32_t


cdef list acceptable_chrom 
cdef list acceptable_nuc
'''
Only take in fragments from regular chromosomes
'''
acceptable_chrom= list(range(1,23))
acceptable_chrom.extend(['X','Y'])
acceptable_chrom = ['chr' + str(chrom) for chrom in acceptable_chrom]
frag_size = 400 #length of the one-hot sequence
acceptable_nuc = list('ACTGN')
dna_encoder = onehot_sequence_encoder(''.join(acceptable_nuc))


cdef str padded_seq(str chrom, str start_str, str end_str , str strand, genome_fa):
    '''
    1. fetch sequence from genome
    2. centering the sequence
    3. pad with Ns on both ends (fill-up to $frag_size$)
    '''
    cdef:
        long start, end, center
        int seq_length, half_padding, padding_base
        str seq

    start, end = long(start_str), long(end_str)
    seq_length = end - start

    if seq_length < frag_size:
        padding_base = frag_size - seq_length
        half_padding = int(padding_base/2)
        seq = genome_fa.fetch(chrom, start, end)
        seq = seq.upper()
        seq = half_padding * 'N' + seq + (half_padding + 1) * 'N'

    else:
        center = (end + start) / 2
        seq = genome_fa.fetch(chrom, 
                                int(center) - int(frag_size/2), 
                                int(center) + int(frag_size/2))

    seq = seq.upper() 
    seq = reverse_complement(seq) if strand == "-" else seq
    return seq[:frag_size]
 


def fetch_trainings(bed_file, fasta):
    '''
    For each record in bed file, extract the sequence, and center it
    fill up both sides to length of (seq_length) with Ns.


    Bed files need these columns:
    1. chrom
    2. start
    3. end
    4. 
    5. 
    6. strand
    7. label: (DNA or RNA)

    return:
        generator: (padded-sequence, label)
    '''
    cdef:
         uint32_t line_count
         str line, chrom, start, end, strand, label
         str seq

    genome_fa = pysam.Fastafile(fasta)
    for line_count, line in enumerate(xopen(bed_file,'r')):
        fields = line.rstrip('\n').split('\t')
        chrom, start, end, strand, label = itemgetter(0,1,2,5,-1)(fields)
        if chrom != 'chrM':
            seq = padded_seq(chrom, start, end, strand, genome_fa)
            yield seq, label


def generate_padded_data(bed_file, fasta):
    '''
    Wrapper for generating one-hot-encoded sequences

    return:
        generator: (one-hot-encoded sequence, label)
    '''
    cdef:
        str seq, na_label
        int label

    for i, (seq, na_label) in enumerate(fetch_trainings(bed_file, fasta)):
        if set(seq).issubset(acceptable_nuc):
            label = 1 if na_label == "DNA" else 0
            yield dna_encoder.transform(seq), label


class data_generator():
    
    def __init__(self, bed_file, fasta, batch_size=1000):
        '''
        Wrapper for generating one-hot-encoded sequences

        return batches with balanced class
        '''
        self.bed = bed_file
        self.fasta = fasta
        self.batch_size = batch_size
        self.half_batch = self.batch_size/2
        self.generator = self.init_generator()

    def init_generator(self):
        return fetch_trainings(self.bed, self.fasta)

    def data_gen(self):
        '''
        Populate reponse tensor and feature tensor with desired batch size
        '''
        cdef:
            uint32_t i
            str seq, na_label
            double random_frac
            int label
            list X, Y
            int sample_num

        X, Y = [], []

        label_counter = defaultdict(int) #make sure classes label is balanced
        while sample_num < self.batch_size:
            try:
                seq, na_label = next(self.generator)
            except StopIteration:
                self.generator = self.init_generator()
                seq, na_label = next(self.generator)

            if set(seq).issubset(acceptable_nuc):
                random_frac = random.random()
                if label_counter[na_label] <= self.half_batch and random_frac >= 0.5:
                    X.append(dna_encoder.transform(seq))
                    label = 1 if na_label == "DNA" else 0
                    Y.append(label)
                    label_counter[na_label] += 1
                    sample_num += 1

        return X, Y


    def __next__(self):
        '''
        generator for Keras fit_generator
        '''
        X, Y = self.data_gen()
        return np.array(X), np.array(Y)


def prediction_generator(test_bed, fa_file, batch_size = 1000):
    '''
    parsing each line of a bed file
    fetch sequence and one-hot encode it

    yeild list whenever batch_size is filled
    '''

    cdef:
        list features, lines 
        int skip = 0
        int sample_in_batch = 0
        str bed_line
        str chrom, start, end, strand
        str seq
        uint32_t frag_count

    features, lines =[], []
    assert(batch_size > 0)
    genome_fa = pysam.Fastafile(fa_file)
    with xopen(test_bed, 'r') as bed:
        for frag_count, bed_line in enumerate(bed):
            fields = bed_line.rstrip('\n').split('\t')
            chrom, start, end, strand = itemgetter(0,1,2,5)(fields)
            seq = padded_seq(chrom, start, end, strand, genome_fa)
            if set(seq).issubset(acceptable_nuc):
                features.append(dna_encoder.transform(seq))
                lines.append(bed_line.strip())
                sample_in_batch += 1
                if sample_in_batch % batch_size == 0 and sample_in_batch > 0:
                    yield(np.array(features), lines)
                    features = []
                    lines = []
            else:
                skip += 1

    if lines: 
        yield(np.array(features), lines)
    
    print('Parsed: {frag_count} fragments\n'\
          'Skipped {skip} fragments with non-standard nucleotides'\
          .format(frag_count = frag_count, skip = skip), 
          file=sys.stderr)
 
