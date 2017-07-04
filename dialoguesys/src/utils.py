# NLU Project
# Description: The script performs the preprocessing of all the data

# Import site-packages libraries
import os
import re
import collections
import pandas
import pickle
import numpy as np
import pandas as pd
import pdb

# Import local modules from the package
from configuration import get_configuration

class Preprocessing():
    def __init__(self, train_path_file ="Training_Shuffled_Dataset.txt",
                 test_path_file = "Validation_Shuffled_Dataset.txt",
                 train_path_file_target ="input_train_triples",
                 test_path_file_target ="input_test_triples",
                 config = None, vocab_path_file = "vocab.pkl",
                 bool_processing = True,
                 triples=True):
        """Constructor: it initilizes the attributes of the class by getting the parameters from the config file"""
        self.train_path_file = train_path_file
        self.test_path_file = test_path_file
        self.train_path_file_target = train_path_file_target
        self.test_path_file_target = test_path_file_target
        self.vocab_path_file = vocab_path_file
        self.batch_size = config.batch_size
        self.data_dir = config.data_dir
        self.vocab_size = config.vocab_size
        self.reverse_encoder_inputs = config.reverse_encoder_inputs
        self.max_unk_count = config.max_unk_count
        # Special vocabulary symbols - we always put them at the start.
        self.pad = r"_PAD"
        self.go = r"_GO"
        self.eos = r"_EOS"
        self.unk = r"_UNK"
        self.start_vocab = [self.pad, self.go, self.eos, self.unk]
        # Regular expressions used to tokenize.
        self.word_split = re.compile(r"([.,!?\"':;)(])")
        self.word_re = re.compile(r"^[-]+[-]+")
        # Processing triples?
        self.triples=triples

    def tokenizer(self, sentence, bool_flat_list = True):
        """Function: Very basic tokenizer: split the sentence into a list of tokens.
        Args:
            sentence: A line from the original file data. Note that for this dataset each line has three sentences
            separated by a tab
            bool_flat_list: option to return a flat list or list of list
        """
        words = []
        if bool_flat_list == True:
            for tab_separated_sentence in sentence.split("\t"):
                words.extend(self.preprocess(tab_separated_sentence))
            return words
        else:
            for tab_separated_sentence in sentence.split("\t"):
                words.append(self.preprocess(tab_separated_sentence))
            return words

    def preprocess(self, sentence):
        """Function: Split the data by space and removes special characters.
         Args:
            sentence: A sentence from the dialog
        """
        sentence = [self.word_split.split(self.word_re.sub(r"",word))for word in
        sentence.lower().strip().split()]
        sentence = [word for sublist in sentence for word in sublist if word]
        return sentence

    def create_vocabulary(self, input_path_file, tokenizer = None):
        """Function: Create vocabulary file (if it does not exist yet) from data file.
        Data file is assumed to contain one sentence per line. Each sentence is
        tokenized and digits are normalized (if normalize_digits is set).
        Vocabulary contains the most-frequent tokens up to max_vocabulary_size.
        We write it to vocabulary_path in a one-token-per-line format, so that later
        token in the first line gets id=0, second line gets id=1, and so on.
        Args:
          input_path_file: data file that will be used to create vocabulary.
          tokenizer: a function to use to tokenize each data sentence; if None, internal tokenizer will be used.
        """
        # Set up the path
        path_vocab = os.path.join(self.data_dir, self.vocab_path_file)
        path_input = os.path.join(self.data_dir, input_path_file)
        # Check for an existing file
        if not os.path.exists(path_vocab):
            print("Creating vocabulary %s from data %s" % (self.vocab_path_file, self.data_dir))
            # Initialize dict and list
            self.vocab = {}
            tokens = []
            # Open the file
            with open(path_input, 'r', newline="\n") as f:
                counter = 0
                for line in f:
                    counter += 1
                    if counter % 50000 == 0:
                        print("Processing line %d" % counter)
                    # Process each line
                    tokens.extend(tokenizer(line) if tokenizer != None else self.tokenizer(line))
                # Generate dictionary by selecting the most common words
                counter = collections.Counter(tokens).most_common(self.vocab_size)
                # Save data for better visualization
                pandas.DataFrame.from_dict(counter).to_csv(os.path.join(self.data_dir, "vocab.csv"))
                # Create list of all the words in the vocabulary with the special tag
                self.vocab = dict(counter)
                vocab_list = self.start_vocab + sorted(self.vocab, key=self.vocab.get, reverse = True)
                # Save vocabulary
                print("Saving vocabulary")
                with open(path_vocab, 'wb') as f:
                    pickle.dump(vocab_list, f)

    def initialize_vocabulary(self):
        """Function: Initialize vocabulary from file.
        We assume the vocabulary is stored one-item-per-line, so a file:
          dog
          cat
        will result in a vocabulary {"dog": 0, "cat": 1}, and this function will
        also return the reversed-vocabulary ["dog", "cat"].
        Args:
          vocabulary_path: path to the file containing the vocabulary.
        Returns:
          a pair: the vocabulary (a dictionary mapping string to integers), and
          the reversed vocabulary (a list, which reverses the vocabulary mapping).
        Raises:
          ValueError: if the provided vocabulary_path does not exist.
        """
        path_vocab = os.path.join(self.data_dir, self.vocab_path_file)
        if os.path.exists(path_vocab):
            with open(os.path.join(path_vocab), 'rb') as f:
                list_vocab = pickle.load(f)
            self.dict_vocab_reverse = dict([(idx, word) for (idx, word) in enumerate(list_vocab)])
            self.dict_vocab = dict((word, idx) for idx, word in self.dict_vocab_reverse.items())
        else:
            raise ValueError("Vocabulary file %s not found.", path_vocab)

    def sentence_to_token_ids(self, sentence, tokenizer=None):
        """Function: Convert a string to list of integers representing token-ids.
        For example, a sentence "I have a dog" may become tokenized into
        ["I", "have", "a", "dog"] and with vocabulary {"I": 1, "have": 2,
        "a": 4, "dog": 7"} this function will return [1, 2, 4, 7].
        Args:
          sentence: the sentence in string format to convert to token-ids.
          tokenizer: a function to use to tokenize each sentence;
            if None, basic_tokenizer will be used.
        Returns:
          a list of integers, the token-ids for the sentence.
        """
        # Initialize list
        list_sentences = []
        # Select tokenizer
        if tokenizer:
            sentences = tokenizer(sentence)
        else:
            sentences = self.tokenizer(sentence, bool_flat_list = False)
        # Convert to integers
        for idx in range(len(sentences)):
            list_sentences.append([self.dict_vocab.get(w, self.dict_vocab.get(self.unk)) for w in sentences[idx]])
        return list_sentences

    def token_ids_to_sentence(self, integer_sentences):
        word_sentences = []
        for integer_sentence in integer_sentences:
            word_sentence = [self.dict_vocab_reverse[token_id] for token_id in integer_sentence]
            word_sentence = " ".join(word_sentence[:-1])
            word_sentences.append(word_sentence)
        return word_sentences

    def data_to_token_ids(self, input_path, target_path, tokenizer=None):
        """Tokenize data file and turn into token-ids using given vocabulary file.
        This function loads data line-by-line from data_path, calls the above
        sentence_to_token_ids, and saves the result to target_path. See comment
        for sentence_to_token_ids on the details of token-ids format.
        Args:
          data_path: path to the data file in one-sentence-per-line format.
          target_path: path where the file with token-ids will be created.
          vocabulary_path: path to the vocabulary file.
          tokenizer: a function to use to tokenize each sentence;
            if None, basic_tokenizer will be used.
          normalize_digits: Boolean; if true, all digits are replaced by 0s.
        """
        # Set up the path
        path_input = os.path.join(self.data_dir, input_path)
        path_target = os.path.join(self.data_dir, target_path)
        # Initialize list
        token_ids = []
        # Tokenize
        print("Tokenizing data in %s" % path_target)
        self.initialize_vocabulary()
        with open(path_input, 'r', newline="\n") as f:
            counter = 0
            for line in f:
                counter += 1
                if counter % 10000 == 0:
                    print("Tokenizing line %d" % counter)
                token_ids.append(self.sentence_to_token_ids(line))
        return token_ids

    def prepare_data(self):
        """Prepare all necessary files that are required for the training.
          Args:
          Returns:
            A tuple of 2 elements:
              (1) list of the numpy token-ids for training data-set
              (2) list of the numpy token-ids for test data-set,
        """
        self.initialize_vocabulary()

        # Set up the path
        path_target_train = os.path.join(self.data_dir, self.train_path_file_target + ".pkl")
        path_target_test = os.path.join(self.data_dir, self.test_path_file_target + ".pkl")

        if not os.path.exists(path_target_train):
            # Create token ids for the training data.
            input_train_path = self.train_path_file
            target_train_path  = self.train_path_file_target
            int_train_input = self.data_to_token_ids(input_train_path, target_train_path)

            # Create raw sequences for encoder and decoder inputs
            training_data = self.create_sequence_data(int_train_input)

            # Save pre-processed data to disk
            with open(path_target_train, 'wb') as f:
                pickle.dump(training_data,f)

        if not os.path.exists(path_target_test):
            # Create token ids for the validation data.
            input_test_path = self.test_path_file
            target_test_path = self.test_path_file_target
            int_test_input =  self.data_to_token_ids(input_test_path, target_test_path)

            # Create raw sequences for encoder and decoder inputs
            test_data = self.create_sequence_data(int_test_input, remove_unk_lines=False)

            # Save pre-processed data to disk
            with open(path_target_test, 'wb') as f:
                pickle.dump(test_data, f)

    def create_sequence_data(self, int_data, remove_unk_lines=True):
        encoder_inputs = []
        decoder_targets = []

        for idx in range(len(int_data)):
            encoder_line = list(reversed(int_data[idx][0])) if self.reverse_encoder_inputs else int_data[idx][0]
            decoder_line = int_data[idx][1]
            if ((not remove_unk_lines) or (encoder_line.count(self.dict_vocab.get(self.unk)) <= self.max_unk_count and
                decoder_line.count(self.dict_vocab.get(self.unk)) <= self.max_unk_count)):
                encoder_inputs.append(encoder_line)
                decoder_targets.append(decoder_line)
            else:
                pass
            if self.triples:
                encoder_line = list(reversed(int_data[idx][1])) if self.reverse_encoder_inputs else int_data[idx][1]
                decoder_line = int_data[idx][2]
                if ((not remove_unk_lines) or (encoder_line.count(self.dict_vocab.get(self.unk)) <= self.max_unk_count or
                    decoder_line.count(self.dict_vocab.get(self.unk)) <= self.max_unk_count)):
                    encoder_inputs.append(encoder_line)
                    decoder_targets.append(decoder_line)
                else:
                    pass

        return encoder_inputs, decoder_targets


def generate_batches(int_data, batch_size, num_epochs, pad_value=0, shuffle=True, time_major=False):
    # int_data is a length 2 list - consisting of list of encoder inputs and corresponding list of decoder targets

    def create_equal_length_sequences(seq_array, pad_value=pad_value, batch_size=batch_size):
        """ Given a numpy array of sequences of unequal lengths, return
            sequences of equal length by padding with pad_value
        """
        max_length = max([len(seq) for seq in seq_array])
        original_seq_len = []
        for idx, seq in enumerate(seq_array):
            original_seq_len.append(len(seq))
            seq_array[idx] = seq + [pad_value]*(max_length - len(seq))
        seq_array = np.vstack(seq_array[:]).astype(np.int)
        return seq_array, original_seq_len

    data_size = len(int_data[0])
    encoder_inputs = np.array(int_data[0])
    decoder_targets = np.array(int_data[1])
    num_batches_per_epoch = int((data_size-1)/batch_size) + 1

    for epoch in range(num_epochs):
        if shuffle: indices = np.random.permutation(np.arange(data_size))
        else: indices = np.arange(data_size)
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            encoder_inputs_batch = encoder_inputs[indices[start_index:end_index]]
            decoder_targets_batch = decoder_targets[indices[start_index:end_index]]
            # Create sequences of equal length by padding
            encoder_inputs_batch, encoder_inputs_length = create_equal_length_sequences(encoder_inputs_batch)
            decoder_targets_batch, decoder_targets_length = create_equal_length_sequences(decoder_targets_batch)
            # Transpose if time_major
            if time_major:
                encoder_inputs_batch = np.transpose(encoder_inputs_batch)
                decoder_targets_batch = np.transpose(decoder_targets_batch)
            yield encoder_inputs_batch, encoder_inputs_length, decoder_targets_batch, decoder_targets_length
