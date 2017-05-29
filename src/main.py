import tensorflow as tf
import pickle
import pdb
from utils import generate_batches
from utils import Preprocessing
from configuration import get_configuration
from configuration import print_configuration
<<<<<<< HEAD
from models.basic_model import BasicModel
from models.seq2seq_model import Seq2SeqModel


# Load configuration and create model
config = get_configuration()
model = Seq2SeqModel(config)

# Pre-process the data
preprocess = Preprocessing(config = config)
preprocess.prepare_data()

# Load input data and train model on the input data
train_data = pickle.load(open((config.data_dir) + '/input_train.pkl', 'rb'))
train_batches = generate_batches(train_data, batch_size=config.batch_size,
                                    num_epochs=config.n_epochs, time_major=True)
model.train(train_batches, verbose=True)
=======
from models.se2seq_model import Seq2seq
from utils import Preprocessing

def main():
    # Setup and get current configuration
    config = get_configuration()
    # Print parameters
    print_configuration()
    # Perform preprocessing
    preprocess = Preprocessing(config = config)
    train_input_encoder, train_input_decoder, \
    test_input_encoder, test_input_decoder, = preprocess.prepare_data()
    # Initialize model class
    model = Seq2seq(config, train_input_encoder, train_input_decoder, test_input_encoder, test_input_decoder)
    model.train(preprocess.num_batches_train,preprocess.num_batches_test)
>>>>>>> 1fa5b7fcace975cc21366420399d09563850d41a

# Evaluate perplexity of trained model on validation data
test_data = pickle.load(open((config.data_dir) + '/input_test.pkl', 'rb'))
# model.infer(test_data)
