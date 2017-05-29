# NLU Project
# Configuration file
# Description: The script setup all the parameters for training and saving the output

# Import libraries
import tensorflow as tf
import os
import time
from pathlib import PurePath

# Define parent directory
project_dir = str(PurePath(__file__).parent.parent)  # root of git-project

# Labels for output and data
label_output = "runs"
label_data = "data"

# Define output directory
timestamp = str(int(time.time()))
output_dir = os.path.abspath(os.path.join(os.path.curdir, label_output, timestamp))

# Setup constant parameters
flags = tf.app.flags

# Define directory parameters
flags.DEFINE_string('output_dir', output_dir, 'The directory where all results are stored')
flags.DEFINE_string('data_dir', os.path.join(project_dir, label_data), 'The directory where all input data are stored')

# Define model parameters
flags.DEFINE_bool('debug', True, 'Run in debug mode')
flags.DEFINE_integer('lr', 0.01, 'Learning rate')
flags.DEFINE_integer('rnn_size', 512, 'Number of hidden units')
flags.DEFINE_integer('embedding_dim', 100,'The dimension of the embedded vectors')  # noqa
flags.DEFINE_string('model_name', 'seq2seq', 'Name of the trained model')
flags.DEFINE_string('vocab_size', 25000, 'Total number of different words')
flags.DEFINE_string('grad_clip', 10, 'Limitation of the gradient')
flags.DEFINE_string('max_seq_length', 300, 'Limitation of the gradient')

# Define training parameters
flags.DEFINE_integer('batch_size', 64, 'Batch size')
flags.DEFINE_integer('n_epochs', 100, 'Number of epochs')

# Define general parameters
flags.DEFINE_integer('summary_every', 10, """generate a summary every `n` step. This is for visualization purposes""")
flags.DEFINE_integer('n_checkpoints_to_keep', 3,'keep maximum `integer` of chekpoint files')
flags.DEFINE_integer('evaluate_every', 1,'evaluate trained model every n step')
flags.DEFINE_integer('save_every', 100, 'store checkpoint every n step')

# Tensorflow Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

# Obtain the current paremeters
def get_configuration():
    global FLAGS
    FLAGS = tf.flags.FLAGS
    FLAGS._parse_flags()
    return FLAGS

# Print the current paramets
def print_configuration():
    print("Parameters: ")
    for attr, value in sorted(FLAGS.__flags.items()):
        print("{}={}".format(attr.upper(), value))
