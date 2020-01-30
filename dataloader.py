import youtokentome
import codecs
import os
import torch
from random import shuffle
from itertools import groupby
from torch.nn.utils.rnn import pad_sequence


class SequenceLoader(object):
    """
    An iterator for loading batches of data into the transformer model.

    For training:

        Each batch contains tokens_in_batch target language tokens (approximately),
        target language sequences of the same length to minimize padding and therefore memory usage,
        source language sequences of very similar (if not the same) lengths to minimize padding and therefore memory usage.
        Batches are also shuffled.

    For validation and testing:

        Each batch contains just a single source-target pair, in the same order as in the files from which they were read.
    """

    def __init__(self, data_folder, source_suffix, target_suffix, split, tokens_in_batch):
        """
        :param data_folder: folder containing the source and target language data files
        :param source_suffix: the filename suffix for the source language files
        :param target_suffix: the filename suffix for the target language files
        :param split: train, or val, or test?
        :param tokens_in_batch: the number of target language tokens in each batch
        """
        self.tokens_in_batch = tokens_in_batch
        self.source_suffix = source_suffix
        self.target_suffix = target_suffix
        assert split.lower() in {"train", "val",
                                 "test"}, "'split' must be one of 'train', 'val', 'test'! (case-insensitive)"
        self.split = split.lower()

        # Is this for training?
        self.for_training = self.split == "train"

        # Load BPE model
        self.bpe_model = youtokentome.BPE(model=os.path.join(data_folder, "bpe.model"))

        # Load data
        with codecs.open(os.path.join(data_folder, ".".join([split, source_suffix])), "r", encoding="utf-8") as f:
            source_data = f.read().split("\n")[:-1]
        with codecs.open(os.path.join(data_folder, ".".join([split, target_suffix])), "r", encoding="utf-8") as f:
            target_data = f.read().split("\n")[:-1]
        assert len(source_data) == len(target_data), "There are a different number of source or target sequences!"
        source_lengths = [len(s) for s in self.bpe_model.encode(source_data, bos=False, eos=False)]
        target_lengths = [len(t) for t in self.bpe_model.encode(target_data, bos=True,
                                                                eos=True)]  # target language sequences have <BOS> and <EOS> tokens
        self.data = list(zip(source_data, target_data, source_lengths, target_lengths))

        # If for training, pre-sort by target lengths - required for itertools.groupby() later
        if self.for_training:
            self.data.sort(key=lambda x: x[3])

        # Create batches
        self.create_batches()

    def create_batches(self):
        """
        Prepares batches for one epoch.
        """

        # If training
        if self.for_training:
            # Group or chunk based on target sequence lengths
            chunks = [list(g) for _, g in groupby(self.data, key=lambda x: x[3])]

            # Create batches, each with the same target sequence length
            self.all_batches = list()
            for chunk in chunks:
                # Sort inside chunk by source sequence lengths, so that a batch would also have similar source sequence lengths
                chunk.sort(key=lambda x: x[2])
                # How many sequences in each batch? Divide expected batch size (i.e. tokens) by target sequence length in this chunk
                seqs_per_batch = self.tokens_in_batch // chunk[0][3]
                # Split chunk into batches
                self.all_batches.extend([chunk[i: i + seqs_per_batch] for i in range(0, len(chunk), seqs_per_batch)])

            # Shuffle batches
            shuffle(self.all_batches)
            self.n_batches = len(self.all_batches)
            self.current_batch = -1
        else:
            # Simply return once pair at a time
            self.all_batches = [[d] for d in self.data]
            self.n_batches = len(self.all_batches)
            self.current_batch = -1

    def __iter__(self):
        """
        Iterators require this method defined.
        """
        return self

    def __next__(self):
        """
        Iterators require this method defined.

        :returns: the next batch, containing:
            source language sequences, a tensor of size (N, encoder_sequence_pad_length)
            target language sequences, a tensor of size (N, decoder_sequence_pad_length)
            true source language lengths, a tensor of size (N)
            true target language lengths, typically the same as decoder_sequence_pad_length as these sequences are bucketed by length, a tensor of size (N)
        """
        # Update current batch index
        self.current_batch += 1
        try:
            source_data, target_data, source_lengths, target_lengths = zip(*self.all_batches[self.current_batch])
        # Stop iteration once all batches are iterated through
        except IndexError:
            raise StopIteration

        # Tokenize using BPE model to word IDs
        source_data = self.bpe_model.encode(source_data, output_type=youtokentome.OutputType.ID, bos=False,
                                            eos=False)
        target_data = self.bpe_model.encode(target_data, output_type=youtokentome.OutputType.ID, bos=True,
                                            eos=True)

        # Convert source and target sequences as padded tensors
        source_data = pad_sequence(sequences=[torch.LongTensor(s) for s in source_data],
                                   batch_first=True,
                                   padding_value=self.bpe_model.subword_to_id('<PAD>'))
        target_data = pad_sequence(sequences=[torch.LongTensor(t) for t in target_data],
                                   batch_first=True,
                                   padding_value=self.bpe_model.subword_to_id('<PAD>'))

        # Convert lengths to tensors
        source_lengths = torch.LongTensor(source_lengths)
        target_lengths = torch.LongTensor(target_lengths)

        return source_data, target_data, source_lengths, target_lengths
