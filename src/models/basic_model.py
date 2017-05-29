# NLU Project
# Main file
# Description: The script loads a model and performs training or predictions

# Import site-package libraries
import os
import tensorflow as tf

class BasicModel():

    def __init__(self, config, train_input_encoder, train_input_decoder,
                test_input_encoder, test_input_decoder):
        """
        This is the initialization, it is done for every model.
        It initializes all the parameters needed for training the neural network.
        """

        # Save the configuration parameters in the class attributes
        self.config = config
        self.model_name = config.model_name
        self.out_dir = config.output_dir
        self.lr = self.config.lr
        self.rnn_size = config.rnn_size
        self.batch_size = config.batch_size
        self.embedding_dim = config.embedding_dim
        self.vocab_size = config.vocab_size
        self.debug = config.debug
        self.grad_clip = config.grad_clip
        self.allow_soft_placement = config.allow_soft_placement
        self.log_device_placement = config.log_device_placement
        self.num_checkpoint = config.log_device_placement
        self.num_epoch = config.n_epochs
        self.max_seq_length = config.max_seq_length

        # Define graph and setup main parameters
        self.graph = tf.Graph()
        with self.graph.as_default():
            session_conf = tf.ConfigProto(
                allow_soft_placement=self.allow_soft_placement,
                log_device_placement=self.log_device_placement,
            )
            self.session = tf.Session(config=session_conf)
            with self.session.as_default():
                with tf.device('/gpu:0'):
                    # Setup training data
                    self.labels_train = train_input_decoder[1]
                    self.encoder_data_train = train_input_encoder[0]
                    self.decoder_data_train = train_input_decoder[0]
                    self.encoder_length_train = train_input_encoder[2]
                    self.decoder_length_train = train_input_decoder[2]

                    # Setup test data
                    self.labels_test = test_input_decoder[1]
                    self.encoder_data_test = test_input_encoder[0]
                    self.decoder_data_test = test_input_decoder[0]
                    self.encoder_length_test = test_input_encoder[2]
                    self.decoder_length_test = test_input_decoder[2]

                    # Builds the graph and session
                    self._build_graph()

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

    def _build_summary(self):
        """
        Returns a summary operation that summarizes data / variables during training.

        The summary_op should be defined here and should be run after
        `self._build_graph()` is called.

        `self._make_summary_op()` will be called automatically in the
        `self.__init__`-method.

        Here's an example implementation:

        with self.graph.as_default():
            tf.summary.scalar('loss_summary', tf.get_variable('loss'))
            tf.summary.scalar('learning_rate', tf.get_variable('lr'))
            # ... add more summaries...

            # merge all summaries generated and return the summary_op
            return tf.summary.merge_all()


        ... at a later point, the actual summary is stored like this:
        self.summarize() and it is typically called in `self.learn_from_epoch`.
        """
        # Summaries for loss and accuracy
        loss_summary = tf.summary.scalar("loss", self.loss)
        acc_summary = tf.summary.scalar("accuracy", self.accuracy)

        # Train Summaries
        self.train_summary_op = tf.summary.merge([loss_summary, acc_summary])
        train_summary_dir = os.path.join(self.out_dir, "summaries", "train")
        self.train_summary_writer = tf.summary.FileWriter(train_summary_dir, self.session.graph)

        # Dev summaries
        self.dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
        dev_summary_dir = os.path.join(self.out_dir, "summaries", "dev")
        self.dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, self.session.graph)

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

    def _save(self):
        """
        This is the model save-function. It is intended to be used within
        `self.learn_from_epoch`, but may of course be used anywhere else.
        The checkpoint is stored in `self.checkpoint_dir`.
        """
        # Checkpoint directory (Tensorflow assumes this directory already exists so we need to create it)
        checkpoint_dir = os.path.abspath(os.path.join(self.out_dir, "checkpoints"))
        self.checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=self.num_checkpoint)

    def restore(self):
        """
        This function restores the trained model.
        """

        raise Exception('Needs to be implemented')

    def train(self, corpus, epochs=100):
        """
        This function trains the model.
        """

