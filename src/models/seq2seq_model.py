import numpy as np
import tensorflow as tf
from models.basic_model import BasicModel
import tensorflow.contrib.seq2seq as seq2seq
import helpers

class Seq2SeqModel(BasicModel):

    def __init__(self, config):
        # Define special symbols from vocabulary
        self.pad = 0
        self.go = 1
        self.eos = 2
        self.unk = 3
        tf.set_random_seed(7)
        super(Seq2SeqModel, self).__init__(config)

    def _build_graph(self, graph):
        with graph.as_default(): # What does this line do?
            session_conf = tf.ConfigProto(
                allow_soft_placement=self.allow_soft_placement,
                log_device_placement=self.log_device_placement,
            )
            sess = tf.Session(config=session_conf)
            with sess.as_default(): # Refactor the code to not use sess.as_default()
                self._init_placeholders()
                self._init_decoder_train_connectors()
                self._init_embeddings()
                self._init_encoder()
                self._init_decoder()
                self._init_optimizer()

                # Initialize all variables
                sess.run(tf.global_variables_initializer())
                sess.graph.finalize()

        return graph, sess

    def _init_placeholders(self):
        """ Everything is time-major
        """
        self.encoder_inputs = tf.placeholder(
            shape=(None, None), # Try not specifying any shape here
            dtype=tf.int32,
            name='encoder_inputs' # Why is this necessary?
        )
        self.encoder_inputs_length = tf.placeholder(
            shape=(None,),
            dtype=tf.int32,
            name='encoder_inputs_length',
        )
        self.decoder_targets = tf.placeholder(
            shape=(None, None),
            dtype=tf.int32,
            name='decoder_targets'
        )
        self.decoder_targets_length = tf.placeholder(
            shape=(None,),
            dtype=tf.int32,
            name='decoder_targets_length',
        )

    def _init_decoder_train_connectors(self):
        with tf.name_scope('DecoderTrainFeeds'):
            sequence_size, batch_size = tf.unstack(tf.shape(self.decoder_targets)) # Why use this line? Why not index/directly pick batch_size?
            go_slice = tf.ones([1, batch_size], dtype=tf.int32) * self.go
            pad_slice = tf.ones([1, batch_size], dtype=tf.int32) * self.pad

            self.decoder_train_inputs = tf.concat([go_slice, self.decoder_targets], axis=0)
            self.decoder_train_length = self.decoder_targets_length + 1

            decoder_train_targets = tf.concat([self.decoder_targets, pad_slice], axis=0)
            decoder_train_targets_eos_mask = tf.one_hot(self.decoder_targets_length,
                                                        sequence_size+1,
                                                        on_value=self.eos, off_value=self.pad,
                                                        dtype=tf.int32)
            decoder_train_targets_eos_mask = tf.transpose(decoder_train_targets_eos_mask, [1, 0]) # Need to understand exact format of tf.transpose function
            decoder_train_targets = tf.add(decoder_train_targets,
                                            decoder_train_targets_eos_mask)
            self.decoder_train_targets = decoder_train_targets

            self.loss_weights = tf.ones([
                batch_size,
                tf.reduce_max(self.decoder_train_length)
            ], dtype=tf.float32, name="loss_weights")

    def _init_embeddings(self):
        with tf.variable_scope("embedding") as scope: # What does this line do? How's this different from name_scope?
            self.embedding_matrix = tf.get_variable( # When do we use get_variable vs. Variable?
                name='embedding_matrix',
                shape=(self.vocab_size, self.embedding_dim),
                initializer=tf.random_uniform_initializer(-1.0, 1.0),
                dtype=tf.float32
            )
            self.encoder_inputs_embedded = tf.nn.embedding_lookup(
                self.embedding_matrix, self.encoder_inputs)
            self.decoder_train_inputs_embedded = tf.nn.embedding_lookup(
                self.embedding_matrix, self.decoder_train_inputs)

    def _init_encoder(self):
        with tf.variable_scope("encoder") as scope:
            _, self.encoder_final_state = tf.nn.dynamic_rnn(
                cell=tf.contrib.rnn.BasicLSTMCell(self.rnn_size),
                inputs=self.encoder_inputs_embedded,
                sequence_length=self.encoder_inputs_length, # How exactly are sequence lengths dealt with? docs say it is not for performance
                time_major=True,
                dtype=tf.float32,
            )

    def _init_decoder(self):
        with tf.variable_scope("decoder") as scope: # Need to understand why we aren't using the dynamic_rnn method here
            def output_fn(outputs):
                return tf.contrib.layers.fully_connected(outputs, self.vocab_size,
                    activation_fn=None, scope=scope)
            decoder_cell = tf.contrib.rnn.BasicLSTMCell(self.rnn_size)

            # Train decoding
            train_helper = seq2seq.TrainingHelper(
                inputs=self.decoder_train_inputs_embedded,
                sequence_length=self.decoder_train_length,
                time_major=True
            )
            train_decoder = seq2seq.BasicDecoder(
                cell=decoder_cell,
                helper=train_helper,
                initial_state=self.encoder_final_state
            )
            self.decoder_outputs_train, _ = seq2seq.dynamic_decode(
                decoder=train_decoder,
                output_time_major=True,
                impute_finished=True,
                scope=scope
            )
            self.decoder_logits_train = output_fn(self.decoder_outputs_train.rnn_output)
            self.decoder_prediction_train = tf.argmax(self.decoder_logits_train, 2)

            # Inference decoding
            scope.reuse_variables()
            inference_helper = seq2seq.GreedyEmbeddingHelper(
                embedding=self.embedding_matrix,
                start_tokens=[self.go]*self.batch_size,
                end_token=self.eos
            )
            inference_decoder = seq2seq.BasicDecoder(
                cell=decoder_cell,
                helper=inference_helper,
                initial_state=self.encoder_final_state
            )
            self.decoder_outputs_inference, _ = seq2seq.dynamic_decode(
                decoder=inference_decoder,
                output_time_major=True,
                impute_finished=True,
                scope=scope
            )
            self.decoder_logits_inference = output_fn(self.decoder_outputs_inference.rnn_output)

    def _init_optimizer(self):
        logits = tf.transpose(self.decoder_logits_train, [1, 0, 2])
        targets = tf.transpose(self.decoder_train_targets, [1, 0])
        self.loss = seq2seq.sequence_loss(logits=logits, targets=targets,
                                          weights=self.loss_weights)
        self.train_op = tf.train.AdamOptimizer().minimize(self.loss) # Possibly implement gradient clipping

    def make_train_inputs(self, input_seq, target_seq):
        inputs_, inputs_length_ = helpers.batch(input_seq)
        targets_, targets_length_ = helpers.batch(target_seq)
        return {
            self.encoder_inputs: inputs_,
            self.encoder_inputs_length: inputs_length_,
            self.decoder_targets: targets_,
            self.decoder_targets_length: targets_length_,
        }

    def train(self, train_batches, verbose=True):
        loss_track = []
        for batch_idx, batch in enumerate(train_batches):
            feed_dict = {
                self.encoder_inputs: batch[0],
                self.encoder_inputs_length: batch[1],
                self.decoder_targets: batch[2],
                self.decoder_targets_length: batch[3]
            }
            _, l = self.session.run([self.train_op, self.loss], feed_dict)
            loss_track.append(l)

            if verbose: print('batch {} loss: {}'.format(batch_idx,
                self.session.run(self.loss, feed_dict)))

            # if batch_idx == 0: break

        return loss_track

    def train_on_copy_task(self, length_from=3, length_to=8,
                           vocab_lower=3, vocab_upper=10,
                           batch_size=64,
                           max_batches=5000,
                           batches_in_epoch=1000,
                           verbose=True):
        batches = helpers.random_sequences(length_from=length_from, length_to=length_to,
                                           vocab_lower=vocab_lower, vocab_upper=vocab_upper,
                                           batch_size=batch_size)
        loss_track = []
        try:
            for batch in range(max_batches+1):
                batch_data = next(batches)
                fd = self.make_train_inputs(batch_data, batch_data)
                _, l = self.session.run([self.train_op, self.loss], fd)
                loss_track.append(l)

                if verbose:
                    if batch == 0 or batch % batches_in_epoch == 0:
                        print('batch {}'.format(batch))
                        print('  minibatch loss: {}'.format(self.session.run(self.loss, fd)))
                        for i, (e_in, dt_pred) in enumerate(zip(
                                fd[self.encoder_inputs].T,
                                self.session.run(self.decoder_prediction_train, fd).T
                            )):
                            print('  sample {}:'.format(i + 1))
                            print('    enc input           > {}'.format(e_in))
                            print('    dec train predicted > {}'.format(dt_pred))
                            if i >= 2:
                                break
                        print()
        except KeyboardInterrupt:
            print('training interrupted')

        return loss_track

    def infer(self, test_data):
        """ Evaluate perplexity of trained model on test set
            test_data comprises of a list of two datasets, the first of
            which are the input line and the second is the output line (all tokenized).
            Each line is represented as a list
        """
        for i in range(len(test_data[0])): # Feed one entry to graph for each row of test data
            feed_dict = {
                self.encoder_inputs: np.reshape(np.array(test_data[0][i]), (-1, 1)),
                self.encoder_inputs_length: np.array([len(test_data[0][i])])
            }
            logits = self.session.run(self.decoder_logits_inference, feed_dict)
            targets = test_data[1][i]
            pdb.set_trace()
