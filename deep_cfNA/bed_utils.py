from operator import itemgetter
from collections import defaultdict
from sequencing_tools.fastq_tools import reverse_complement, \
                    onehot_sequence_encoder
import pysam
import random
import numpy as np
import sys
from sequencing_tools.io_tools import xopen


'''
Only take in fragments from regular chromosomes
'''
acceptable_chrom = list(range(1,23))
acceptable_chrom.extend(['X','Y'])
acceptable_chrom = ['chr' + str(chrom) for chrom in acceptable_chrom]
frag_size = 400
acceptable_nuc = list('ACTGN')
dna_encoder = onehot_sequence_encoder(''.join(acceptable_nuc))


def padded_seq(chrom, start, end ,strand, genome_fa):
    start, end = int(start), int(end)
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
    genome_fa = pysam.Fastafile(fasta)
    for i, line in enumerate(open(bed_file)):
        fields = line.rstrip('\n').split('\t')
        chrom, start, end, strand, label = itemgetter(0,1,2,5,-1)(fields)
        if chrom != 'chrM':
            seq = padded_seq(chrom, start, end, strand, genome_fa)
            yield seq, label


def generate_padded_data(bed_file, fasta):
    '''
    Wrapper for generating one-hot-encoded sequences
    '''
    for i, (seq, label) in enumerate(get_padded_seq(bed_file, fasta)):
        if set(seq).issubset(acceptable_nuc):
            label = 1 if label == "DNA" else 0
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
        self.generator = get_padded_seq(self.bed, self.fasta)

    def data_gen(self):
        '''
        Populate reponse vector and feature array with desired batch size
        '''
        X, Y = [], []

        label_counter = defaultdict(int)
        for i in range(self.batch_size):
            try:
                seq, label = next(self.generator)
            except StopIteration:
                self.generator = get_padded_seq(self.bed, self.fasta)
                seq, label = next(self.generator)

            if set(seq).issubset(acceptable_nuc):
                if label_counter[label] <= self.batch_size/2 and random.random() >= 0.5:
                    X.append(dna_encoder.transform(seq))
                    label = 1 if label == "DNA" else 0
                    Y.append(label)
                    label_counter[label] += 1
        return X, Y


    def __next__(self):
        '''
        generator for Keras fit_generator
        '''
        X, Y = self.data_gen()
        return np.array(X), np.array(Y)


def prediction_generator(test_bed, fa_file, batch_size = 1000):
    assert(batch_size > 0)
    features = []
    lines = []
    sample_in_batch = 0
    skip = 0
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
    
    print('Skipped %i fragments with non-standard nucleotides' %skip, file=sys.stderr)
 
