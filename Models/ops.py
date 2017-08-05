import tensorflow as tf

def fully_connected(input_, output_nodes, name, stddev=0.02):
    with tf.variable_scope(name):
        input_shape = input_.get_shape()
        input_nodes = input_shape[-1]
        w = tf.get_variable('w', [input_nodes, output_nodes], 
            initializer=tf.truncated_normal_initializer(stddev=0.02))
        biases = tf.get_variable('b', [output_nodes], 
            initializer=tf.constant_initializer(0.0))
        res = tf.matmul(input_, w) + biases
        return res


# 1d CONVOLUTION WITH DILATION
def conv1d(input_, output_channels, 
    dilation = 1, filter_width = 1, causal = False, 
    name = "dilated_conv"):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [1, filter_width, input_.get_shape()[-1], output_channels ],
            initializer=tf.truncated_normal_initializer(stddev=0.02))
        b = tf.get_variable('b', [output_channels ],
           initializer=tf.constant_initializer(0.0))

        if causal:
            padding = [[0, 0], [(filter_width - 1) * dilation, 0], [0, 0]]
            padded = tf.pad(input_, padding)
            input_expanded = tf.expand_dims(padded, dim = 1)
            out = tf.nn.atrous_conv2d(input_expanded, w, rate = dilation, padding = 'VALID') + b
        else:
            input_expanded = tf.expand_dims(input_, dim = 1)
            out = tf.nn.atrous_conv2d(input_expanded, w, rate = dilation, padding = 'SAME') + b

        return tf.squeeze(out, [1])