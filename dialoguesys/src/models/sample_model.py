import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from models.basic_model import BasicModel


class TestModel(BasicModel):

    def _build_graph(self, graph):
        with graph.as_default():
            global_step = tf.Variable(0, trainable=False, name='global_step')

            x = tf.placeholder(tf.float32, shape=[None, 784], name='x')
            y_ = tf.placeholder(tf.float32, shape=[None, 10], name='y_')
            W = tf.Variable(tf.zeros([784, 10]))
            b = tf.Variable(tf.zeros([10]), name='b')

            y = tf.matmul(x, W) + b

            cross_entropy = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_),
                name='cross_entropy')

            optimizer = tf.train.GradientDescentOptimizer(0.5, name='optimizer')  # noqa
            train_op = optimizer.minimize(cross_entropy, global_step=global_step, name='train_op')  # noqa

            init_op = tf.global_variables_initializer()

        return (graph, init_op)

    def learn_from_epoch(self, epoch):
        with self.graph.as_default() as graph:
            train_op = graph.get_operation_by_name('train_op')

            def _train_step(t):
                feed = self.yield_batch()
                self.session.run([train_op],  # noqa
                    feed_dict=feed)
                if t % self.config['summary_every'] == 0:
                    self.summarize(feed)
                if t % self.config['save_every'] == 0:
                    self.save()

            for t in range(1000):
                _train_step(t)

    def yield_batch(self):
        batch = self.mnist.train.next_batch(100)
        # for the x:0: https://github.com/tensorflow/tensorflow/issues/3378

        return {'x:0': batch[0], 'y_:0': batch[1]}

    def _make_summary_op(self):
        with self.graph.as_default() as graph:
            tf.summary.scalar('summary_cross_entropy',
                              graph.get_tensor_by_name('cross_entropy:0'))
            # ... add more tf.summaries here

            return tf.summary.merge_all()

    def _init(self):
        self.summary_op = self._make_summary_op()
        self.mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
