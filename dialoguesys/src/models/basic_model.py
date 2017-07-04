# NLU Project
# Main file
# Description: The script loads a model and performs training or predictions

# Import site-package libraries
import os
import tensorflow as tf
import time

class BasicModel():

    def __init__(self, config):
        """
        This is the initialization, it is done for every model.
        It initializes all the parameters needed for training the neural network.
        """

        # Save the configuration parameters in the class attributes
        self.config = config
        self.model_name = config.model_name
        self.out_dir = config.output_dir
        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)
        self.learning_rate_decay_factor = config.decay_learning_rate
        self.learning_rate_init = config.lr
        self.lr_decay_sample_size = config.lr_decay_sample_size
        self.rnn_size = config.rnn_size
        self.batch_size = config.batch_size
        self.embedding_dim = config.embedding_dim
        self.vocab_size = config.vocab_size
        self.beam_width = config.beam_width
        self.debug = config.debug
        self.grad_clip = config.grad_clip
        self.dropout_keep_prob = config.dropout_keep_prob
        self.allow_soft_placement = config.allow_soft_placement
        self.log_device_placement = config.log_device_placement
        self.summary_every = config.summary_every
        self.save_every = config.save_every
        self.evaluate_every = config.evaluate_every
        self.train_summary_dir = os.path.join(self.out_dir, "summaries", "train")
        self.eval_summary_dir = os.path.join(self.out_dir, "summaries", "eval")
        self.n_checkpoints_to_keep = config.n_checkpoints_to_keep

        # Builds the graph and session
        self.graph, self.session = self._build_graph(tf.Graph())

    def _build_graph(self, graph):
        """
        This is where the actual graph is constructed. Returns the tuple
        `(graph, init_op)` where `graph` is the computational graph and
        `init_op` is the operation that initializes variables in the graph.

        Notice that summarizing is done in `self._make_summary_op`. Also
        notice that saving is done in `self.save`. That means,
        `self._build_graph` does not need to implement summarization and
        saving.

        Example:
        with graph.as_default():
            input_x = tf.placeholder(tf.int64, [64,100])
            input_y = tf.placeholder(tf.int64, [64,1])
            with tf.variable_scope('rnn'):
                W = tf.Variable(
                    tf.random_uniform([64,100]), -.1, .1, name='weights')
                b = tf.Variable(tf.zeros([64,1]))
                ...
            with tf.variable_scope('optimize'):
                tvars = tf.trainable_variables()
                optimizer = tf.train.AdamOptimizer()
                grads, tvars = zip(*optimizer.compute_gradients(loss, tvars))

                train_op = optimizer.apply_gradients(
                    zip(grads, tvars), global_step=tf.Variable(0, trainable=False))  # noqa

            init_op = tf.global_variables_initializer()
        return (graph, init_op)
        """

        raise Exception('Needs to be implemented')

    def infer(self):
        """
        This method is used for coming with new predictions. It assumes that
        the model is already trained.
        """

        raise Exception('Needs to be implemented')

    def evaluate(self, epoch):
        """
        Evaluates current loss function given the validation dataset
        """

        raise Exception('Needs to be implemented')

    def summarize(self, feed_dict):
        """
        Writes a summary to `self.summary_dir`. This is useful during training
        to see how training is going (e.g the value of the loss function)

        This method assumes that `self._make_summary_op` has been called. It
        may be a single operation or a list of operations
        """

        raise Exception('Needs to be implemented')

    def save(self):
        """
        This is the model save-function. It is intended to be used within
        `self.learn_from_epoch`, but may of course be used anywhere else.
        The checkpoint is stored in `self.checkpoint_dir`.
        """
        # Checkpoint directory (Tensorflow assumes this directory already exists so we need to create it)
        checkpoint_dir = os.path.abspath(os.path.join(self.out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

    def restore(self):
        """
        This function restores the trained model.
        """

        raise Exception('Needs to be implemented')

    def train(self, corpus, epochs=100):
        """
        This function trains the model.
        """
