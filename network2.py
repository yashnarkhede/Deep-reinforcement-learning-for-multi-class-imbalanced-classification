import tensorflow as tf


class QNetwork:
    def __init__(self, config):
        size = 32
        channel = 3
        self.config = config
        self.n_class = len(self.config.new_class)
        tf.compat.v1.reset_default_graph()
        with tf.compat.v1.variable_scope('input'):
            self.state = tf.compat.v1.placeholder(shape=[None, size, size, channel], dtype=tf.float32)
            self.learning_rate = tf.compat.v1.placeholder(dtype=tf.float32)
            self.target_q = tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.float32)
            self.reward = tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.float32)
            self.action = tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.int32)
            self.terminal = tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.float32)
            self.target_soft_update = tf.compat.v1.placeholder(dtype=tf.float32)
        with tf.compat.v1.variable_scope('main_net'):
            self.q_mnet = self._build_network()
        with tf.compat.v1.variable_scope('target_net'):
            self.q_tnet = self._build_network()

        main_variables = tf.compat.v1.get_collection(key=tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope="main_net")
        target_variables = tf.compat.v1.get_collection(key=tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope="target_net")

        self.update_target = [t.assign((1 - self.target_soft_update) * t + self.target_soft_update * m)
                              for t, m in zip(target_variables, main_variables)]
        self.q_wrt_a = tf.expand_dims(tf.gather_nd(self.q_mnet, self.action, batch_dims=1), axis=1)
        self.target = self.reward + (1 - self.terminal) * self.config.gamma * self.target_q
        self.loss = tf.compat.v1.losses.huber_loss(self.target, self.q_wrt_a)
        self.train_op = tf.compat.v1.train.AdamOptimizer(self.learning_rate).minimize(self.loss, var_list=main_variables)

    def _build_network(self):
        x = tf.compat.v1.layers.conv2d(self.state, 32, 5, strides=2, activation=tf.nn.relu)
        x = tf.compat.v1.layers.conv2d(x, 32, 5, strides=2, activation=tf.nn.relu)
        x = tf.compat.v1.layers.flatten(x)
        x = tf.compat.v1.layers.dense(x, 256, activation=tf.nn.relu)
        a = tf.compat.v1.layers.dense(x, self.n_class)
        v = tf.compat.v1.layers.dense(x, 1)
        q = v + a - tf.reduce_mean(input_tensor=a, axis=1, keepdims=True)
        return q