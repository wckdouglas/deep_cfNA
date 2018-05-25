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
from libc.stdlib cimport rand, RAND_MAX
from libc.stdint cimport uint32_t


cpdef double random_generator():
    '''
    generate random fraction between 0 and 1
    '''
    return rand()/RAND_MAX * 1.

'''
Only take in fragments from regular chromosomes
'''
cdef:
    list acceptable_chrom 
    list acceptable_nuc

acceptable_chrom= list(range(1,23))
acceptable_chrom.extend(['X','Y'])
acceptable_chrom = ['chr' + str(chrom) for chrom in acceptable_chrom]
frag_size = 400
acceptable_nuc = list('ACTGN')
dna_encoder = onehot_sequence_encoder(''.join(acceptable_nuc))


cdef str padded_seq(str chrom, str start_str, str end_str , str strand, genome_fa):
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
 


def get_padded_seq(bed_file, fasta):
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
    '''
    cdef:
         uint32_t line_count
         str line, chrom, start, end, strand, label
         str seq

    genome_fa = pysam.Fastafile(fasta)
    for line_count, line in enumerate(open(bed_file)):
        fields = line.rstrip('\n').split('\t')
        chrom, start, end, strand, label = itemgetter(0,1,2,5,-1)(fields)
        if chrom != 'chrM':
            seq = padded_seq(chrom, start, end, strand, genome_fa)
            yield seq, label


def generate_padded_data(bed_file, fasta):
    '''
    Wrapper for generating one-hot-encoded sequences
    '''
    cdef:
        str seq, na_label
        int label

    for i, (seq, na_label) in enumerate(get_padded_seq(bed_file, fasta)):
        if set(seq).issubset(acceptable_nuc):
            label = 1 if na_label == "DNA" else 0
            yield dna_encoder.transform(seq), label


class data_generator():
    
    def __init__(self, bed_file, fasta, batch_size=1000):
        '''
        Wrapper for generating one-hot-encoded sequences

        return batches
        '''
        self.bed = bed_file
        self.fasta = fasta
        self.batch_size = batch_size
        self.half_batch = self.batch_size/2
        self.generator = get_padded_seq(self.bed, self.fasta)

    def data_gen(self):
        '''
        Populate reponse vector and feature array with desired batch size
        '''
        cdef:
            uint32_t i
            str seq, na_label
            double random_frac
            int label
            list X, Y

        X, Y = [], []

        label_counter = defaultdict(int)
        for i in range(self.batch_size):
            try:
                seq, na_label = next(self.generator)
            except StopIteration:
                self.generator = get_padded_seq(self.bed, self.fasta)
                seq, na_label = next(self.generator)

            if set(seq).issubset(acceptable_nuc):
                random_frac = random.random()
                if label_counter[na_label] <= self.half_batch and random_frac >= 0.5:
                    X.append(dna_encoder.transform(seq))
                    label = 1 if na_label == "DNA" else 0
                    Y.append(label)
                    label_counter[na_label] += 1
        return X, Y


    def __next__(self):
        '''
        generator for Keras fit_generator
        '''
        X, Y = self.data_gen()
        return np.array(X), np.array(Y)


def prediction_generator(test_bed, fa_file, batch_size = 1000):
    cdef:
        list features, lines 
        int skip = 0
        int sample_in_batch = 0
        str bed_line
        str chrom, start, end, strand
        str seq

    features, lines =[], []
    assert(batch_size > 0)
    genome_fa = pysam.Fastafile(fa_file)
    with xopen(test_bed, 'r') as bed:
        for bed_line in bed:
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
    
    print('Skipped %i fragments with non-standard nucleotides' %skip, 
          file=sys.stderr)
 
