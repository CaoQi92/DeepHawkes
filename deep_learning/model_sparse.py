import sys
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell


class SDPP(object):
    def __init__(self, config, sess, node_embed):
        
        self.n_sequences = config.n_sequences
        self.learning_rate = config.learning_rate
        self.emb_learning_rate = config.emb_learning_rate
        self.training_iters = config.training_iters
        self.sequence_batch_size = config.sequence_batch_size
        self.batch_size = config.batch_size
        self.display_step = config.display_step
        self.n_time_interval = config.n_time_interval
        self.embedding_size = config.embedding_size
        self.n_input = config.n_input
        self.n_steps = config.n_steps
        self.n_hidden_gru = config.n_hidden_gru
        self.n_hidden_dense1 = config.n_hidden_dense1
        self.n_hidden_dense2 = config.n_hidden_dense2
        self.scale1 = config.l1
        self.scale2 = config.l2
        self.scale = config.l1l2
        if config.activation == "tanh":
            self.activation = tf.tanh
        else:
            self.activation = tf.nn.relu
        self.max_grad_norm = config.max_grad_norm
        self.initializer = tf.random_normal_initializer(stddev=config.stddev)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.regularizer = tf.contrib.layers.l1_l2_regularizer(self.scale1, self.scale2)
        self.dropout_prob = config.dropout_prob
        self.sess = sess
        self.node_vec = node_embed
        self.name = "deephawkes"
        
        
        self.build_input()
        self.build_var()
        self.pred = self.build_model()
        
        truth = self.y
        # Define loss and optimizer
        cost = tf.reduce_mean(tf.pow(self.pred - truth, 2)) + self.scale*tf.add_n([self.regularizer(var) for var in tf.trainable_variables()])
        error = tf.reduce_mean(tf.pow(self.pred - truth, 2))
        tf.summary.scalar("error", error)
        
        var_list1 = [var for var in tf.trainable_variables() if not 'embedding' in var.name]
        var_list2 = [var for var in tf.trainable_variables() if 'embedding' in var.name]
        opt1 = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        opt2 = tf.train.AdamOptimizer(learning_rate=self.emb_learning_rate)
        grads = tf.gradients(cost, var_list1 + var_list2)
        grads1 = [tf.clip_by_norm(grad, self.max_grad_norm) for grad in grads[:len(var_list1)]]
        grads2 = [tf.clip_by_norm(grad, self.max_grad_norm) for grad in grads[len(var_list1):]]
        train_op1 = opt1.apply_gradients(zip(grads1, var_list1))
        train_op2 = opt2.apply_gradients(zip(grads2, var_list2))
        train_op = tf.group(train_op1, train_op2)
        self.cost = cost
        self.error = error
        self.train_op = train_op
        
        init_op = tf.initialize_all_variables()
        self.sess.run(init_op)
        
    
    def build_input(self):
        self.x = tf.placeholder(tf.int32, shape=[None, self.n_steps], name="x")
        # (total_number of sequence,n_steps)
        self.x_indict = tf.placeholder(tf.int64,shape=[None,3])
        # (total number of sequence,dim_index)
        self.y = tf.placeholder(tf.float32, [None, 1], name="y")
        self.time_interval_index = tf.placeholder(tf.float32, [None,self.n_time_interval], name="time")
        # (total_number of sequence,n_time_interval)
        self.rnn_index = tf.placeholder(tf.float32, [None,self.n_steps], name="rnn_index")
        # (total_number of sequence,n_steps)
        # self.rnn_length = tf.placeholder(tf.float32, [None,1], name="rnn_length")
    def build_var(self):
        with tf.variable_scope(self.name) as scope:
            with tf.variable_scope('embedding'):
                self.embedding = tf.get_variable('embedding', initializer=tf.constant(self.node_vec, dtype=tf.float32))
            with tf.variable_scope('BiGRU'):
                self.gru_fw_cell = rnn_cell.GRUCell(2*self.n_hidden_gru)
            with tf.variable_scope('SumPooling'):
                self.time_weight = tf.get_variable('time_weight', initializer=self.initializer([self.n_time_interval]), dtype=tf.float32)
		#self.time_weight = tf.mul(self.time_weight_temp,self.time_weight_temp)
            with tf.variable_scope('dense'):
                self.weights = {
                    'dense1': tf.get_variable('dense1_weight', initializer=self.initializer([2 * self.n_hidden_gru,
                                                                                        self.n_hidden_dense1])),
                   'dense2': tf.get_variable('dense2_weight', initializer=self.initializer([self.n_hidden_dense1,
                                                                                       self.n_hidden_dense2])),
                    'out': tf.get_variable('out_weight', initializer=self.initializer([self.n_hidden_dense2, 1]))
                }
                self.biases = {
                    'dense1': tf.get_variable('dense1_bias', initializer=self.initializer([self.n_hidden_dense1])),
                   'dense2': tf.get_variable('dense2_bias', initializer=self.initializer([self.n_hidden_dense2])),
                    'out': tf.get_variable('out_bias', initializer=self.initializer([1]))
                }
                
                
    
    def build_model(self):
        with tf.device('/gpu:0'):
            with tf.variable_scope('deephawkes') as scope:
                with tf.variable_scope('embedding'):
                    x_vector = tf.nn.dropout(tf.nn.embedding_lookup(self.embedding, self.x), 
                                             self.dropout_prob)
                    # (total_number of sequence, n_steps, n_input)
                with tf.variable_scope('RNN'):
                    x_vector = tf.transpose(x_vector, [1,0,2])
                    # (n_steps, total_number of sequence, n_input)
                    x_vector = tf.reshape(x_vector, [-1, self.n_input])
                    # (n_steps*total_number of sequence, n_input)


                    # Split to get a list of 'n_steps' tensors of shape (n_sequences*batch_size, n_input)
                    x_vector = tf.split(0, self.n_steps, x_vector)

                    outputs, _ = rnn.rnn(self.gru_fw_cell, x_vector, dtype=tf.float32)

                    hidden_states = tf.transpose(tf.pack(outputs), [1, 0, 2])
                    # (total_number of sequence, n_steps, n_hidden_gru)

                    # filter according to the length
                    hidden_states = tf.reshape(hidden_states,[-1,2*self.n_hidden_gru])
                    #   (total_number of sequence*n_step, 2*n_hidden_gru)

                    rnn_index = tf.reshape(self.rnn_index,[-1,1])
                    #   (total_number of sequence*n_step,1)


                    hidden_states = tf.mul(rnn_index,hidden_states)
                    #   (total_number of sequence*n_step, 2*n_hidden_gru)

                    hidden_states = tf.reshape(hidden_states,[-1,self.n_steps,2*self.n_hidden_gru])
                    #   (total_number of sequence,n_step,2*n_hidden_gru)

                    hidden_states = tf.reduce_sum(hidden_states, reduction_indices=[1])
                    #   (total_number of sequence,2*n_hidden_gru)

                with tf.variable_scope('SumPooling'):
                    # sumpooling

                    time_weight = tf.reshape(self.time_weight,[-1,1])
                    #   (n_time_interval,1)
                    #   time_interval_index    (total_number of sequence,n_time_interval)
                    time_weight = tf.matmul(self.time_interval_index,time_weight)
                    #   (total_number of sequence,1)

                    hidden_graph_value = tf.mul(time_weight,hidden_states)
                    #   (total_number of sequence,2*n_hidden_gru)

                    hidden_graph_value = tf.reshape(hidden_graph_value,[-1])
                    #   (total_number of sequence*2*n_hidden_gru)

                    hidden_graph = tf.SparseTensor(indices = self.x_indict, values=hidden_graph_value,
                                                   shape=[self.batch_size, self.n_sequences, 2 * self.n_hidden_gru])

                    hidden_graph = tf.sparse_reduce_sum(hidden_graph, axis=1)
                    # self.batch_size, 2 * self.n_hidden_gru
        
                with tf.variable_scope('dense'):
                    dense1 = self.activation(tf.add(tf.matmul(hidden_graph, self.weights['dense1']), self.biases['dense1']))
                    dense2 = self.activation(tf.add(tf.matmul(dense1, self.weights['dense2']), self.biases['dense2']))
                    pred = self.activation(tf.add(tf.matmul(dense2, self.weights['out']), self.biases['out']))
                    print pred.get_shape()
                return pred
        
    def train_batch(self, x, x_indict,y, time_interval_index,rnn_index):
        #merged = tf.summary.merge_all()
        _,time_weight = self.sess.run([self.train_op,self.time_weight],
                                                          feed_dict={self.x: x, self.x_indict: x_indict, self.y: y,
                                                                     self.time_interval_index:time_interval_index,
                                                                     self.rnn_index:rnn_index})
        # print rnn_state
        return time_weight
    def get_embedding(self, x, x_indict,y, time_interval_index,rnn_index):
        embedding = self.sess.run([self.embedding],
                                                          feed_dict={self.x: x, self.x_indict: x_indict, self.y: y,
                                                                     self.time_interval_index:time_interval_index,
                                                                     self.rnn_index:rnn_index})
        # print rnn_state
        return embedding
    def get_error(self, x, x_indict,y, time_interval_index,rnn_index):
        return self.sess.run(self.error, feed_dict={self.x: x, self.x_indict: x_indict, self.y: y,
                                                                     self.time_interval_index:time_interval_index,
                                                                     self.rnn_index:rnn_index})
    def predict(self, x, x_indict,y, time_interval_index,rnn_index):
        return self.sess.run(self.pred, feed_dict={self.x: x, self.x_indict: x_indict, self.y: y,
                                                                     self.time_interval_index:time_interval_index,
                                                                     self.rnn_index:rnn_index})

