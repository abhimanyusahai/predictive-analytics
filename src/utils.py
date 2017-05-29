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
                 train_path_file_target ="input_train",
                 test_path_file_target ="input_test",
                 config = None, vocab_path_file = "vocab.pkl",
                 bool_processing = True):
        """Constructor: it initilizes the attributes of the class by getting the parameters from the config file"""
        self.train_path_file = train_path_file
        self.test_path_file = test_path_file
        self.train_path_file_target = train_path_file_target
        self.test_path_file_target = test_path_file_target
        self.vocab_path_file = vocab_path_file
        self.batch_size = config.batch_size
        self.data_dir = config.data_dir
        self.vocab_size = config.vocab_size
        self.max_seq_length = config.max_seq_length

        # Special vocabulary symbols - we always put them at the start.
        self.pad = r"_PAD"
        self.go = r"_GO"
        self.eos = r"_EOS"
        self.unk = r"_UNK"
        self.start_vocab = [self.pad, self.go, self.eos, self.unk]

        # Regular expressions used to tokenize.
        self.word_split = re.compile(r"([.,!?\"':;)(])")
        self.word_re = re.compile(r"^[-]+[-]+")

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
        # Set up the path
        path_target_train = os.path.join(self.data_dir, self.train_path_file_target + ".pkl")
        path_target_test = os.path.join(self.data_dir, self.test_path_file_target + ".pkl")

        if not os.path.exists(path_target_train) or not os.path.exists(path_target_test):
            # Create vocabularies of the appropriate sizes.
            self.create_vocabulary(self.train_path_file)

            # Create token ids for the training data.
            input_train_path = self.train_path_file
            target_train_path  = self.train_path_file_target
            int_train_input = self.data_to_token_ids(input_train_path, target_train_path)

            # Create token ids for the validation data.
            input_test_path = self.test_path_file
            target_test_path = self.test_path_file_target
            int_test_input =  self.data_to_token_ids(input_test_path, target_test_path)

            # Create raw sequences for encoder and decoder inputs
            training_data = self.create_sequence_data(int_train_input)
            test_data = self.create_sequence_data(int_test_input)

            # Save pre-processed data to disk
            with open(path_target_train, 'wb') as f:
                pickle.dump(training_data,f)
            with open(path_target_test, 'wb') as f:
                pickle.dump(test_data, f)
<<<<<<< HEAD
=======
        else:
            # Load data
            with open(path_target_train, 'rb') as f:
                training_data = pickle.load(f)
            with open(path_target_test, 'rb') as f:
                test_data = pickle.load(f)

        # Convert list into a numpy array - training data
        train_encoder = pd.DataFrame(training_data[0]).fillna(value=0).astype(int).values
        train_decoder = pd.DataFrame(training_data[1]).fillna(value=0).astype(int).values
        train_length_encoder = np.array(training_data[2], dtype=int)
        train_length_decoder = np.array(training_data[3], dtype=int)

        # Convert list into a numpy array - test data
        test_encoder = pd.DataFrame(test_data[0]).fillna(value=0).astype(int).values
        test_decoder = pd.DataFrame(test_data[1]).fillna(value=0).astype(int).values
        test_length_encoder = np.array(test_data[2], dtype=int)
        test_length_decoder = np.array(test_data[3], dtype=int)

        # Calculate number of batches
        np.random.seed(13)
        self.num_batches_train= int(train_encoder.shape[0] / self.batch_size) + 1
        self.num_batches_test = int(test_encoder.shape[0] / self.batch_size) + 1

        # Add random sentence to complete batches - training
        num_add_train = ((self.num_batches_train) * self.batch_size) - train_encoder.shape[0]
        rand_indices = np.random.choice(train_encoder.shape[0], num_add_train )
        train_encoder = np.vstack((train_encoder, train_encoder[rand_indices, :]))
        train_decoder = np.vstack((train_decoder, train_decoder[rand_indices, :]))
        train_length_encoder = np.hstack((train_length_encoder, train_length_encoder[rand_indices]))
        train_length_decoder = np.hstack((train_length_decoder, train_length_decoder[rand_indices]))

        # Printing maximum length
        print("Maximum lenght encoder {}".format(str(np.max(train_length_encoder))))
        print("Maximum lenght decoder {}".format(str(np.max(train_length_decoder))))
        print("Average lenght encoder {}".format(str(np.mean(train_length_encoder))))
        print("Average lenght decoder {}".format(str(np.mean(train_length_decoder))))
>>>>>>> 1fa5b7fcace975cc21366420399d09563850d41a

    def create_sequence_data(self, int_data):
        encoder_inputs = [sublist[0] for sublist in int_data]
        encoder_inputs.extend([sublist[1] for sublist in int_data])
        decoder_targets = [sublist[1] for sublist in int_data]
        decoder_targets.extend([sublist[2] for sublist in int_data])
        return encoder_inputs, decoder_targets


<<<<<<< HEAD
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
=======
        # Split data into batches - training
        train_x_batch_encoder, train_y_batch_encoder, train_lengh_encoder = \
        self.generate_batches(train_encoder,
        train_length_encoder,self.num_batches_train)
        train_x_batch_decoder, train_y_batch_decoder, train_lengh_decoder = \
        self.generate_batches(train_decoder,
        train_length_decoder,self.num_batches_train)
        print(train_x_batch_encoder.shape)
        print(train_x_batch_decoder.shape)

        # Split data into batches - test
        test_x_batch_encoder, test_y_batch_encoder, test_lengh_encoder = \
        self.generate_batches(test_encoder, test_length_encoder,self.num_batches_test)
        test_x_batch_decoder, test_y_batch_decoder, test_lengh_decoder = \
        self.generate_batches(test_decoder, test_length_decoder,self.num_batches_test)

        # Concatenate arrays into a list
        training_data_encoder = [train_x_batch_encoder, train_y_batch_encoder, train_lengh_encoder]
        training_data_decoder = [train_x_batch_decoder, train_y_batch_decoder, train_lengh_decoder]
        test_data_encoder = [test_x_batch_encoder, test_y_batch_encoder, test_lengh_encoder]
        test_data_decoder = [test_x_batch_decoder, test_y_batch_decoder, test_lengh_decoder]

        # Return output
        return training_data_encoder, training_data_decoder, test_data_encoder, test_data_decoder

    def create_full_batch(self, int_data):
        """It separates the triplets into two input training data and labels.
           Args:
               int_data: it is a list of list that maps ids to words. Every element is an integer
           Returns:
                list_decoder: the training data used for the decoder
                list_encoder: the training data used for the encoder
                list_length_encoder: the lenght of each sentence in the encoder
                list_length_decorder: the lenght of each sentence in the decoder
            Note :
                It is very import to make a deep copy of the list otherwise it will make reference to the list. Use [:]
           """
        list_encoder = []
        list_decoder = []
        list_length_encoder = []
        list_length_decoder = []
        for idx_batch in range(len(int_data)):
            for idx_sentence in range(len(int_data[idx_batch]) - 1):
                # Remove long sentences
                if len(int_data[idx_batch][idx_sentence]) <= self.max_seq_length \
                    and len(int_data[idx_batch][idx_sentence + 1]) <= self.max_seq_length:
                    # Separate data into encoder and decoder input
                    list_encoder.append(int_data[idx_batch][idx_sentence][::-1])
                    list_decoder.append(int_data[idx_batch][idx_sentence + 1][:])
                    # Add special tags
                    list_decoder[-1].insert(0,self.dict_vocab.get(self.go))
                    list_decoder[-1].append(self.dict_vocab.get(self.eos))
                    # Measure the lenght or each list
                    list_length_encoder.append(len(list_encoder[-1]))
                    list_length_decoder.append(len(list_decoder[-1]))
        return list_encoder, list_decoder, list_length_encoder, list_length_decoder

    def generate_batches(self, int_data, data_length_sentences, num_batches):
        """It separates the triplets into two input training data and labels.
           Args:
               int_data: numpy array to split
               data_length_sentences: numpy array to split
           Returns:
                list_decoder: the training data used for the decoder

           """
        # Create batches
        x_batch = int_data
        y_batch = np.copy(int_data)
        y_batch[:, :-1] = x_batch[:, 1:]
        y_batch[:, -1] = x_batch[:, 0]

        # Operation for training or test datasets
        x_batch = np.split(x_batch, num_batches)
        y_batch = np.split(y_batch, num_batches)
        lenght_sentences = np.split(data_length_sentences, num_batches)

        # Return output and convert to array
        return np.asarray(x_batch), np.asarray(y_batch), np.asarray(lenght_sentences)
>>>>>>> 1fa5b7fcace975cc21366420399d09563850d41a
