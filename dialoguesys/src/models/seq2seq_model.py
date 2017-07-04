import numpy as np
import tensorflow as tf
from models.basic_model import BasicModel
import tensorflow.contrib.seq2seq as seq2seq
from tensorflow.python.layers import core as layers_core
from operator import mul
from functools import reduce
import helpers
import os
import pdb
import datetime

class Seq2SeqModel(BasicModel):

    def __init__(self, config):
        # Define special symbols from vocabulary
        self.pad = 0
        self.go = 1
        self.eos = 2
        self.unk = 3
        self.num_special_symbols = 4
        self.max_decoder_seq_length = 500
        tf.set_random_seed(7)
        super(Seq2SeqModel, self).__init__(config)

    def _build_graph(self, graph):
        with graph.as_default(): # What does this line do?
            self.effective_vocab_size = self.vocab_size + self.num_special_symbols
            session_conf = tf.ConfigProto(
                allow_soft_placement=self.allow_soft_placement,
                log_device_placement=self.log_device_placement,
            )
            sess = tf.Session(config=session_conf)
            with sess.as_default(): # Refactor the code to not use sess.as_default()
                with tf.device('/gpu:0'):
                    self._init_placeholders()
                    self._init_decoder_train_connectors()
                    self._init_embeddings()
                    self._init_encoder()
                    self._init_decoder()
                    self._init_optimizer()
                    self._init_others()

                # Initialize variables and finaize graph
                sess.run(tf.global_variables_initializer())
                sess.graph.finalize()

        return graph, sess

    def _init_placeholders(self):
        """ Everything is batch-major
        """
        self.encoder_inputs = tf.placeholder(
            shape=(None, None),
            dtype=tf.int32,
            name='encoder_inputs'
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
        # For inference
        self.decoder_start_tokens = tf.placeholder(
            shape=(None,),
            dtype=tf.int32,
            name='decoder_start_tokens'
        )
        self.inference_sequence = tf.placeholder(
            shape=(None, None),
            dtype=tf.int32,
            name='inference_sequence'
        )
        self.inference_sequence_length = tf.placeholder(
            dtype=tf.int32,
            name='inference_sequence_length'
        )
        # For beam search decoding
        self.encoder_inputs_beam = tf.placeholder(
            shape=(500, 1),
            dtype=tf.int32,
            name='encoder_inputs_beam'
        )
        self.encoder_inputs_length_beam = tf.placeholder(
            shape=(1,),
            dtype=tf.int32,
            name='encoder_inputs_length_beam',
        )

    def _init_decoder_train_connectors(self):
        with tf.name_scope('DecoderTrainFeeds'):
            batch_size, sequence_size = tf.unstack(tf.shape(self.decoder_targets)) # Why use this line? Why not index/directly pick batch_size?
            go_slice = tf.ones([batch_size, 1], dtype=tf.int32) * self.go
            pad_slice = tf.ones([batch_size, 1], dtype=tf.int32) * self.pad

            self.decoder_train_inputs = tf.concat([go_slice, self.decoder_targets], axis=1)
            self.decoder_train_length = self.decoder_targets_length + 1

            decoder_train_targets = tf.concat([self.decoder_targets, pad_slice], axis=1)
            decoder_train_targets_eos_mask = tf.one_hot(self.decoder_targets_length,
                                                        sequence_size+1,
                                                        on_value=self.eos, off_value=self.pad,
                                                        dtype=tf.int32)
            decoder_train_targets = tf.add(decoder_train_targets,
                                            decoder_train_targets_eos_mask)
            self.decoder_train_targets = decoder_train_targets

            self.loss_weights = tf.to_float(tf.sequence_mask(self.decoder_train_length,
                tf.minimum(self.max_decoder_seq_length,
                    tf.reduce_max(self.decoder_train_length))))

    def _init_embeddings(self):
        with tf.variable_scope("embedding") as scope: # What does this line do? How's this different from name_scope?
            self.embedding_matrix = tf.get_variable( # When do we use get_variable vs. Variable?
                name='embedding_matrix',
                shape=(self.effective_vocab_size, self.embedding_dim),
                initializer=tf.random_uniform_initializer(-1.0, 1.0),
                dtype=tf.float32
            )
            self.encoder_inputs_embedded = tf.nn.embedding_lookup(
                self.embedding_matrix, self.encoder_inputs)
            self.decoder_train_inputs_embedded = tf.nn.embedding_lookup(
                self.embedding_matrix, self.decoder_train_inputs)
            # For beam decoding
            self.encoder_inputs_embedded_beam = tf.nn.embedding_lookup(
                self.embedding_matrix, self.encoder_inputs_beam)

    def _init_encoder(self):
        self.encoder_cell = tf.contrib.rnn.BasicLSTMCell(self.rnn_size)
        with tf.variable_scope("encoder") as scope:
            _, self.encoder_final_state = tf.nn.dynamic_rnn(
                cell=self.encoder_cell,
                inputs=self.encoder_inputs_embedded,
                sequence_length=self.encoder_inputs_length, # How exactly are sequence lengths dealt with? docs say it is not for performance
                time_major=False,
                dtype=tf.float32,
            )
            # For beam search decoding
            # scope.reuse_variables()
            # _, self.encoder_final_state_beam = tf.nn.dynamic_rnn(
            #     cell=self.encoder_cell,
            #     inputs=self.encoder_inputs_embedded_beam,
            #     sequence_length=self.encoder_inputs_length_beam, # How exactly are sequence lengths dealt with? docs say it is not for performance
            #     time_major=False,
            #     dtype=tf.float32,
            # )

    def _init_decoder(self):
        self.decoder_cell = tf.contrib.rnn.BasicLSTMCell(self.rnn_size)
        with tf.variable_scope("decoder") as scope: # Need to understand why we aren't using the dynamic_rnn method here
            output_layer = layers_core.Dense(units=self.effective_vocab_size, activation=None)

            # Train decoding
            train_helper = seq2seq.TrainingHelper(
                inputs=self.decoder_train_inputs_embedded,
                sequence_length=self.decoder_train_length,
                time_major=False
            )
            train_decoder = seq2seq.BasicDecoder(
                cell=self.decoder_cell,
                helper=train_helper,
                initial_state=self.encoder_final_state
            )
            self.decoder_outputs_train, _, _ = seq2seq.dynamic_decode(
                decoder=train_decoder,
                output_time_major=False,
                impute_finished=True,
                maximum_iterations=self.max_decoder_seq_length,
                scope=scope
            )
            self.decoder_logits_train = output_layer.apply(self.decoder_outputs_train.rnn_output)
            self.decoder_prediction_train = tf.argmax(self.decoder_logits_train, 2)

            # Greedy decoding
            scope.reuse_variables()
            greedy_helper = seq2seq.GreedyEmbeddingHelper(
                embedding=self.embedding_matrix,
                start_tokens=self.decoder_start_tokens,
                end_token=self.eos
            )
            greedy_decoder = seq2seq.BasicDecoder(
                cell=self.decoder_cell,
                helper=greedy_helper,
                initial_state=self.encoder_final_state,
                output_layer=output_layer
            )
            self.decoder_outputs_inference, _, _ = seq2seq.dynamic_decode(
                decoder=greedy_decoder,
                output_time_major=False,
                impute_finished=True,
                maximum_iterations=self.max_decoder_seq_length, # Figure out a better way of setting this
                scope=scope
            )
            self.decoder_prediction_inference = tf.argmax(self.decoder_outputs_inference.rnn_output, 2)

            # Beam Search Decoding
            # self.decoder_start_tokens_beam = tf.reshape(self.decoder_start_tokens, (1,))
            # beam_decoder = seq2seq.BeamSearchDecoder(
            #     cell=decoder_cell,
            #     embedding=self.embedding_matrix,
            #     start_tokens=self.decoder_start_tokens_beam,
            #     end_token=self.eos,
            #     initial_state=self.encoder_final_state_beam,
            #     beam_width=self.beam_width,
            #     output_layer=output_layer
            # )
            # self.decoder_outputs_beam, _, _ = seq2seq.dynamic_decode(
            #     decoder=beam_decoder,
            #     output_time_major=True,
            #     impute_finished=True,
            #     maximum_iterations=self.max_decoder_seq_length, # Figure out a better way of setting this
            #     scope=scope
            # )

    def _init_optimizer(self):
        with tf.name_scope("optimizer"):
            tvars = tf.trainable_variables()
            logits = self.decoder_logits_train
            targets = tf.slice(
                self.decoder_train_targets,
                begin=[0, 0],
                size=[-1, tf.minimum(self.max_decoder_seq_length,
                    tf.reduce_max(self.decoder_train_length))]
            )
            self.learning_rate = tf.Variable(self.learning_rate_init, trainable=False,
                dtype=tf.float32)
            self.learning_rate_decay_op = self.learning_rate.assign(self.learning_rate *
                self.learning_rate_decay_factor)
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            self.loss = seq2seq.sequence_loss(logits=logits, targets=targets,
                                              weights=self.loss_weights)
            optimizer = tf.train.AdamOptimizer(self.learning_rate)
            grads, tvars = zip(*optimizer.compute_gradients(self.loss, tvars))
            grads, _ = tf.clip_by_global_norm(grads, self.grad_clip)
            self.train_op = optimizer.apply_gradients(zip(grads, tvars),
                global_step=self.global_step)

            # For evaluation
            self.loss_per_line = seq2seq.sequence_loss(logits=logits, targets=targets,
                                weights=self.loss_weights, average_across_batch=False)
            self.perplexity_per_line = tf.exp(self.loss_per_line)

    def _init_others(self):
        # Init saver
        self.saver = tf.train.Saver(max_to_keep=self.n_checkpoints_to_keep)
        # Init summaries
        self.loss_summary = tf.summary.scalar("loss", self.loss)
        self.train_summary_writer = tf.summary.FileWriter(self.train_summary_dir)
        self.eval_summary_writer = tf.summary.FileWriter(self.eval_summary_dir)

        # Calculate total number of parameters
        total_params = 0
        for var in tf.trainable_variables():
            shape = var.get_shape()
            total_params = total_params + reduce(mul, shape).value
        print("Total number of parameters: {:.2f} million".format(total_params/1e6))

    def make_train_inputs(self, input_seq, target_seq):
        inputs_, inputs_length_ = helpers.batch(input_seq)
        targets_, targets_length_ = helpers.batch(target_seq)
        return {
            self.encoder_inputs: inputs_,
            self.encoder_inputs_length: inputs_length_,
            self.decoder_targets: targets_,
            self.decoder_targets_length: targets_length_,
        }

    def train(self, train_batches, eval_batches=None, verbose=True, decay_learning_rate=False):
        loss_track = []
        evaluation_line = ""
        for batch_idx, batch in enumerate(train_batches):
            feed_dict = {
                self.encoder_inputs: batch[0],
                self.encoder_inputs_length: batch[1],
                self.decoder_targets: batch[2],
                self.decoder_targets_length: batch[3]
            }
            _, loss, step, summary = self.session.run([self.train_op, self.loss,
                self.global_step, self.loss_summary], feed_dict)
            loss_track.append(loss)

            # If average of last 20 losses has gone up, decay the training rate
            if decay_learning_rate:
                if len(loss_track) > self.lr_decay_sample_size + 1:
                    if (np.mean(loss_track[-1:-(self.lr_decay_sample_size+1):-1]) >
                        np.mean(loss_track[-2:-(self.lr_decay_sample_size+2):-1])):
                        print("Decaying learning rate")
                        self.learning_rate_decay_op

            # Print training loss summary
            if step % self.summary_every == 0:
                self.train_summary_writer.add_summary(summary, step)
                if verbose:
                    time_str = datetime.datetime.now().isoformat()
                    print('{}: batch {} loss: {}'.format(time_str, step, loss))

            # Save graph
            if step % self.save_every == 0:
                self.saver.save(self.session, self.out_dir + '/model', global_step=step)

            # Evaluate the loss on validation set periodically
            if step % self.evaluate_every == 0:
                loss, perplexity = self.evaluate(eval_batches, load_trained_model=False)
                evaluation_line += "step {}: loss={}, perplexity={}\n".format(step, loss, perplexity)
                with(open(self.out_dir + '/validation_loss.csv', 'w')) as f:
                    f.write(evaluation_line)

    def train_on_copy_task(self, length_from=3, length_to=8,
                           vocab_lower=3, vocab_upper=10,
                           batch_size=64,
                           max_batches=5000,
                           batches_in_epoch=1000,
                           verbose=True):
        """ Feed small inputs into the seq2seq graph to ensure it is functioning
            correctly. Only used in the early stages of the project for debugging
        """
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

    def evaluate(self, eval_batches, load_trained_model=True, model_dir=None, model_file=None):
        """ Evaluate perplexity of trained model on eval set
            Batches of eval set are sent to this function
            (just like in training). Perplexity is calculated
            for each line in the batch and added to the log
        """
        # Load saved variables from training if required
        if load_trained_model:
            checkpoint_file = model_dir + model_file
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            saver.restore(self.session, checkpoint_file)

        perplexity_log = np.array([])
        for batch_idx, batch in enumerate(eval_batches):
            print("Evaluating perplexity on batch {}".format(batch_idx+1))
            feed_dict = {
                self.encoder_inputs: batch[0],
                self.encoder_inputs_length: batch[1],
                self.decoder_targets: batch[2],
                self.decoder_targets_length: batch[3]
            }
            perplexity_per_line = self.session.run(self.perplexity_per_line, feed_dict)
            perplexity_log = np.concatenate((perplexity_log, perplexity_per_line))

        # Reshape the perplexity log into desired output format
        if load_trained_model:
            col1 = perplexity_log[:int(perplexity_log.shape[0]/2)]
            col2 = perplexity_log[int(perplexity_log.shape[0]/2):]
            perplexity_log = np.c_[col1, col2]
            np.savetxt(model_dir + '/perplexity_log.csv', perplexity_log, delimiter=',')

        # Return overall loss and perplexity
        overall_avg_loss = np.mean(np.log(perplexity_log))
        overall_avg_perplexity = np.exp(overall_avg_loss)

        return overall_avg_loss, overall_avg_perplexity

    def infer(self, checkpoint_file, test_data, verbose=True):
        """ Input a list of integer tokenized sentences and get corresponding
            list of conversational outputs
            test_data is a list of lists, where each sublist represents an input
            integer tokenized sentence
        """
        # Load saved variables from training
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(self.session, checkpoint_file)
        predicted_outputs = []

        for i in range(len(test_data)): # Feed one entry to graph for each row of test data
            if verbose: print("Predicting output for sentence {}".format(i+1))
            feed_dict = {
                self.encoder_inputs: np.reshape(np.array(test_data[i]), (-1, 1)),
                self.encoder_inputs_length: np.array([len(test_data[i])]),
                self.decoder_start_tokens: np.array([self.go]),
                self.inference_sequence: np.reshape(np.array(test_data[i]), (-1, 1)),
                self.inference_sequence_length: len(test_data[i])
            }
            prediction = list(np.reshape(self.session.run(self.decoder_prediction_inference,
                feed_dict), -1))
            predicted_outputs.append(prediction)

        return predicted_outputs

# Model with dropout + attention
class Seq2SeqModelAttention(BasicModel):

    def __init__(self, config):
        # Define special symbols from vocabulary
        self.pad = 0
        self.go = 1
        self.eos = 2
        self.unk = 3
        self.num_special_symbols = 4
        self.max_decoder_seq_length = 500
        tf.set_random_seed(7)
        super(Seq2SeqModelAttention, self).__init__(config)

    def _build_graph(self, graph):
        with graph.as_default(): # What does this line do?
            self.effective_vocab_size = self.vocab_size + self.num_special_symbols
            session_conf = tf.ConfigProto(
                allow_soft_placement=self.allow_soft_placement,
                log_device_placement=self.log_device_placement,
            )
            sess = tf.Session(config=session_conf)
            with sess.as_default(): # Refactor the code to not use sess.as_default()
                with tf.device('/gpu:0'):
                    self._init_placeholders()
                    self._init_decoder_train_connectors()
                    self._init_embeddings()
                    self._init_encoder()
                    self._init_decoder()
                    self._init_optimizer()
                    self._init_others()

                # Initialize variables and finaize graph
                sess.run(tf.global_variables_initializer())
                sess.graph.finalize()

        return graph, sess

    def _init_placeholders(self):
        """ Everything is batch-major
        """
        self.encoder_inputs = tf.placeholder(
            shape=(None, None),
            dtype=tf.int32,
            name='encoder_inputs'
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
        # For inference
        self.decoder_start_tokens = tf.placeholder(
            shape=(None,),
            dtype=tf.int32,
            name='decoder_start_tokens'
        )
        self.inference_sequence = tf.placeholder(
            shape=(None, None),
            dtype=tf.int32,
            name='inference_sequence'
        )
        self.inference_sequence_length = tf.placeholder(
            dtype=tf.int32,
            name='inference_sequence_length'
        )
        # For beam search decoding
        self.encoder_inputs_beam = tf.placeholder(
            shape=(500, 1),
            dtype=tf.int32,
            name='encoder_inputs_beam'
        )
        self.encoder_inputs_length_beam = tf.placeholder(
            shape=(1,),
            dtype=tf.int32,
            name='encoder_inputs_length_beam',
        )
        # Add drouput
        self.keep_prob = tf.placeholder(tf.float32, name="keep_prob")

    def _init_decoder_train_connectors(self):
        with tf.name_scope('DecoderTrainFeeds'):
            batch_size, sequence_size = tf.unstack(tf.shape(self.decoder_targets)) # Why use this line? Why not index/directly pick batch_size?
            go_slice = tf.ones([batch_size, 1], dtype=tf.int32) * self.go
            pad_slice = tf.ones([batch_size, 1], dtype=tf.int32) * self.pad

            self.decoder_train_inputs = tf.concat([go_slice, self.decoder_targets], axis=1)
            self.decoder_train_length = self.decoder_targets_length + 1

            decoder_train_targets = tf.concat([self.decoder_targets, pad_slice], axis=1)
            decoder_train_targets_eos_mask = tf.one_hot(self.decoder_targets_length,
                                                        sequence_size+1,
                                                        on_value=self.eos, off_value=self.pad,
                                                        dtype=tf.int32)
            decoder_train_targets = tf.add(decoder_train_targets,
                                            decoder_train_targets_eos_mask)
            self.decoder_train_targets = decoder_train_targets

            self.loss_weights = tf.to_float(tf.sequence_mask(self.decoder_train_length,
                tf.minimum(self.max_decoder_seq_length,
                    tf.reduce_max(self.decoder_train_length))))

    def _init_embeddings(self):
        with tf.variable_scope("embedding") as scope: # What does this line do? How's this different from name_scope?
            self.embedding_matrix = tf.get_variable( # When do we use get_variable vs. Variable?
                name='embedding_matrix',
                shape=(self.effective_vocab_size, self.embedding_dim),
                initializer=tf.random_uniform_initializer(-1.0, 1.0),
                dtype=tf.float32
            )
            self.encoder_inputs_embedded = tf.nn.embedding_lookup(
                self.embedding_matrix, self.encoder_inputs)
            self.decoder_train_inputs_embedded = tf.nn.embedding_lookup(
                self.embedding_matrix, self.decoder_train_inputs)
            # For beam decoding
            self.encoder_inputs_embedded_beam = tf.nn.embedding_lookup(
                self.embedding_matrix, self.encoder_inputs_beam)

    def _init_encoder(self):
        self.encoder_cell = tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.LSTMCell(self.rnn_size),
                                                             output_keep_prob = self.keep_prob)
        with tf.variable_scope("encoder") as scope:
            self.encoder_outputs, self.encoder_final_state = tf.nn.dynamic_rnn(
                cell=self.encoder_cell,
                inputs=self.encoder_inputs_embedded,
                sequence_length=self.encoder_inputs_length, # How exactly are sequence lengths dealt with? docs say it is not for performance
                time_major=False,
                dtype=tf.float32,
            )
            # For beam search decoding
            # scope.reuse_variables()
            # _, self.encoder_final_state_beam = tf.nn.dynamic_rnn(
            #     cell=self.encoder_cell,
            #     inputs=self.encoder_inputs_embedded_beam,
            #     sequence_length=self.encoder_inputs_length_beam, # How exactly are sequence lengths dealt with? docs say it is not for performance
            #     time_major=False,
            #     dtype=tf.float32,
            # )

    def _init_decoder(self):
        lstm_decoder = tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.LSTMCell(self.rnn_size),
                                                             output_keep_prob = self.keep_prob)
        attention_mechanism = tf.contrib.seq2seq.LuongAttention(self.rnn_size, self.encoder_outputs,
            name='LuongAttention')
        self.decoder_cell = tf.contrib.seq2seq.AttentionWrapper(lstm_decoder, attention_mechanism,
            attention_layer_size=self.rnn_size, name="AttentionWrapper")
        batch_size = tf.shape(self.encoder_inputs)[0]
        attn_zero = self.decoder_cell.zero_state(batch_size=batch_size, dtype=tf.float32)
        init_state = attn_zero.clone(cell_state=self.encoder_final_state)
        with tf.variable_scope("decoder") as scope: # Need to understand why we aren't using the dynamic_rnn method here
            output_layer = layers_core.Dense(units=self.effective_vocab_size, activation=None)

            # Train decoding
            train_helper = seq2seq.TrainingHelper(
                inputs=self.decoder_train_inputs_embedded,
                sequence_length=self.decoder_train_length,
                time_major=False
            )
            train_decoder = seq2seq.BasicDecoder(
                cell=self.decoder_cell,
                helper=train_helper,
                initial_state=init_state
            )
            self.decoder_outputs_train, _, _ = seq2seq.dynamic_decode(
                decoder=train_decoder,
                output_time_major=False,
                impute_finished=True,
                maximum_iterations=self.max_decoder_seq_length,
                scope=scope
            )
            self.decoder_logits_train = output_layer.apply(self.decoder_outputs_train.rnn_output)
            self.decoder_prediction_train = tf.argmax(self.decoder_logits_train, 2)

            # Greedy decoding
            scope.reuse_variables()
            greedy_helper = seq2seq.GreedyEmbeddingHelper(
                embedding=self.embedding_matrix,
                start_tokens=self.decoder_start_tokens,
                end_token=self.eos
            )
            greedy_decoder = seq2seq.BasicDecoder(
                cell=self.decoder_cell,
                helper=greedy_helper,
                initial_state=init_state,
                output_layer=output_layer
            )
            self.decoder_outputs_inference, _, _ = seq2seq.dynamic_decode(
                decoder=greedy_decoder,
                output_time_major=False,
                impute_finished=True,
                maximum_iterations=self.max_decoder_seq_length, # Figure out a better way of setting this
                scope=scope
            )
            self.decoder_prediction_inference = tf.argmax(self.decoder_outputs_inference.rnn_output, 2)

            # Beam Search Decoding
            # self.decoder_start_tokens_beam = tf.reshape(self.decoder_start_tokens, (1,))
            # beam_decoder = seq2seq.BeamSearchDecoder(
            #     cell=decoder_cell,
            #     embedding=self.embedding_matrix,
            #     start_tokens=self.decoder_start_tokens_beam,
            #     end_token=self.eos,
            #     initial_state=self.encoder_final_state_beam,
            #     beam_width=self.beam_width,
            #     output_layer=output_layer
            # )
            # self.decoder_outputs_beam, _, _ = seq2seq.dynamic_decode(
            #     decoder=beam_decoder,
            #     output_time_major=True,
            #     impute_finished=True,
            #     maximum_iterations=self.max_decoder_seq_length, # Figure out a better way of setting this
            #     scope=scope
            # )

    def _init_optimizer(self):
        with tf.name_scope("optimizer"):
            tvars = tf.trainable_variables()
            logits = self.decoder_logits_train
            targets = tf.slice(
                self.decoder_train_targets,
                begin=[0, 0],
                size=[-1, tf.minimum(self.max_decoder_seq_length,
                    tf.reduce_max(self.decoder_train_length))]
            )
            self.learning_rate = tf.Variable(self.learning_rate_init, trainable=False,
                dtype=tf.float32)
            self.learning_rate_decay_op = self.learning_rate.assign(self.learning_rate *
                self.learning_rate_decay_factor)
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            self.loss = seq2seq.sequence_loss(logits=logits, targets=targets,
                                              weights=self.loss_weights)
            optimizer = tf.train.AdamOptimizer(self.learning_rate)
            grads, tvars = zip(*optimizer.compute_gradients(self.loss, tvars))
            grads, _ = tf.clip_by_global_norm(grads, self.grad_clip)
            self.train_op = optimizer.apply_gradients(zip(grads, tvars),
                global_step=self.global_step)

            # For evaluation
            self.loss_per_line = seq2seq.sequence_loss(logits=logits, targets=targets,
                                weights=self.loss_weights, average_across_batch=False)
            self.perplexity_per_line = tf.exp(self.loss_per_line)

    def _init_others(self):
        # Init saver
        self.saver = tf.train.Saver(max_to_keep=self.n_checkpoints_to_keep)
        # Init summaries
        self.loss_summary = tf.summary.scalar("loss", self.loss)
        self.train_summary_writer = tf.summary.FileWriter(self.train_summary_dir)
        self.eval_summary_writer = tf.summary.FileWriter(self.eval_summary_dir)

        # Calculate total number of parameters
        total_params = 0
        for var in tf.trainable_variables():
            shape = var.get_shape()
            total_params = total_params + reduce(mul, shape).value
        print("Total number of parameters: {:.2f} million".format(total_params/1e6))

    def make_train_inputs(self, input_seq, target_seq):
        inputs_, inputs_length_ = helpers.batch(input_seq)
        targets_, targets_length_ = helpers.batch(target_seq)
        return {
            self.encoder_inputs: inputs_,
            self.encoder_inputs_length: inputs_length_,
            self.decoder_targets: targets_,
            self.decoder_targets_length: targets_length_,
        }

    def train(self, train_batches, eval_batches=None, verbose=True, decay_learning_rate=False):
        loss_track = []
        evaluation_line = ""
        for batch_idx, batch in enumerate(train_batches):
            feed_dict = {
                self.encoder_inputs: batch[0],
                self.encoder_inputs_length: batch[1],
                self.decoder_targets: batch[2],
                self.decoder_targets_length: batch[3],
                self.keep_prob: self.dropout_keep_prob
            }
            _, loss, step, summary = self.session.run([self.train_op, self.loss,
                self.global_step, self.loss_summary], feed_dict)
            loss_track.append(loss)

            # If average of last 20 losses has gone up, decay the training rate
            if decay_learning_rate:
                if len(loss_track) > self.lr_decay_sample_size + 1:
                    if (np.mean(loss_track[-1:-(self.lr_decay_sample_size+1):-1]) >
                        np.mean(loss_track[-2:-(self.lr_decay_sample_size+2):-1])):
                        print("Decaying learning rate")
                        self.learning_rate_decay_op

            # Print training loss summary
            if step % self.summary_every == 0:
                self.train_summary_writer.add_summary(summary, step)
                if verbose:
                    time_str = datetime.datetime.now().isoformat()
                    print('{}: batch {} loss: {}'.format(time_str, step, loss))

            # Save graph
            if step % self.save_every == 0:
                self.saver.save(self.session, self.out_dir + '/model', global_step=step)

            # Evaluate the loss on validation set periodically
            if step % self.evaluate_every == 0:
                loss, perplexity = self.evaluate(eval_batches, load_trained_model=False)
                evaluation_line += "step {}: loss={}, perplexity={}\n".format(step, loss, perplexity)
                with(open(self.out_dir + '/validation_loss.csv', 'w')) as f:
                    f.write(evaluation_line)

    def train_on_copy_task(self, length_from=3, length_to=8,
                           vocab_lower=3, vocab_upper=10,
                           batch_size=64,
                           max_batches=5000,
                           batches_in_epoch=1000,
                           verbose=True):
        """ Feed small inputs into the seq2seq graph to ensure it is functioning
            correctly. Only used in the early stages of the project for debugging
        """
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

    def evaluate(self, eval_batches, load_trained_model=True, model_dir=None, model_file=None):
        """ Evaluate perplexity of trained model on eval set
            Batches of eval set are sent to this function
            (just like in training). Perplexity is calculated
            for each line in the batch and added to the log
        """
        # Load saved variables from training if required
        if load_trained_model:
            checkpoint_file = model_dir + model_file
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            saver.restore(self.session, checkpoint_file)

        perplexity_log = np.array([])
        for batch_idx, batch in enumerate(eval_batches):
            print("Evaluating perplexity on batch {}".format(batch_idx+1))
            feed_dict = {
                self.encoder_inputs: batch[0],
                self.encoder_inputs_length: batch[1],
                self.decoder_targets: batch[2],
                self.decoder_targets_length: batch[3],
                self.keep_prob: 1.0
            }
            perplexity_per_line = self.session.run(self.perplexity_per_line, feed_dict)
            perplexity_log = np.concatenate((perplexity_log, perplexity_per_line))

        # Reshape the perplexity log into desired output format
        if load_trained_model:
            col1 = perplexity_log[:int(perplexity_log.shape[0]/2)]
            col2 = perplexity_log[int(perplexity_log.shape[0]/2):]
            perplexity_log = np.c_[col1, col2]
            np.savetxt(model_dir + '/perplexity_log.csv', perplexity_log, delimiter=',')

        # Return overall loss and perplexity
        overall_avg_loss = np.mean(np.log(perplexity_log))
        overall_avg_perplexity = np.exp(overall_avg_loss)

        return overall_avg_loss, overall_avg_perplexity

    def infer(self, checkpoint_file, test_data, verbose=True):
        """ Input a list of integer tokenized sentences and get corresponding
            list of conversational outputs
            test_data is a list of lists, where each sublist represents an input
            integer tokenized sentence
        """
        # Load saved variables from training
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(self.session, checkpoint_file)
        predicted_outputs = []

        for i in range(len(test_data)): # Feed one entry to graph for each row of test data
            if verbose: print("Predicting output for sentence {}".format(i+1))
            feed_dict = {
                self.encoder_inputs: np.reshape(np.array(test_data[i]), (1, -1)),
                self.encoder_inputs_length: np.array([len(test_data[i])]),
                self.decoder_start_tokens: np.array([self.go]),
                self.inference_sequence: np.reshape(np.array(test_data[i]), (1, -1)),
                self.inference_sequence_length: len(test_data[i]),
                self.keep_prob: 1.0
            }
            prediction = list(np.reshape(self.session.run(self.decoder_prediction_inference,
                feed_dict), -1))
            predicted_outputs.append(prediction)

        return predicted_outputs
