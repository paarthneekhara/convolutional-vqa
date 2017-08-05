import tensorflow as tf
import numpy as np
import ops
from tensorflow.contrib.layers import layer_norm


def encoder_bytenet(source_sentence, options, train = True):
    # MASKING TILL THE ACTUAL SENTENCE LENGTH IN THE BATCH
    # EXPECTS THE SOURCE SENTENCE TO BE ZERO PADDED TOWARDS THE END
    source_mask = tf.sign(tf.reduce_max( tf.abs(source_sentence), 2 ))
    length = tf.cast( tf.reduce_sum(source_mask, 1), tf.int32)

    source_mask = tf.expand_dims(source_mask, dim = 2)
    source_sentence_reshaped = tf.reshape(source_sentence, [-1, int(source_sentence.get_shape()[2]) ])
    source_embedding = tf.nn.relu( ops.fully_connected( source_sentence_reshaped, 2 * options['residual_channels'], 
        name = "source_sentence_embedding" ) )
    
    source_embedding = tf.reshape(source_embedding, 
        [-1, int(source_sentence.get_shape()[1]), 2 * options['residual_channels'] ])

    curr_input = source_embedding
    for layer_no, dilation in enumerate(options['encoder_dilations']):
        curr_input = curr_input * source_mask
        layer_output = _byetenet_residual_block(curr_input, dilation, layer_no, options, source_mask, train)
        curr_input = layer_output

    processed_output = tf.nn.relu( ops.conv1d(tf.nn.relu(layer_output), 
            options['residual_channels'], 
            name = 'encoder_post_processing') )
    processed_output = processed_output * source_mask
    
    last_seq_element = _last_seq_element(processed_output, length)
    
    return {
        "length" : length,
        "encoder_output" : processed_output,
        "last_seq_element" : last_seq_element,
    }

def _last_seq_element(output, length):
    batch_size = tf.shape(output)[0]
    max_length = tf.shape(output)[1]
    out_size = int(output.get_shape()[2])
    index = tf.range(0, batch_size) * max_length + (length - 1)
    flat = tf.reshape(output, [-1, out_size])
    relevant = tf.gather(flat, index)
    return relevant

def _byetenet_residual_block(input_, dilation, layer_no, options, source_mask, train = True):
        # input_ = layer_norm(input_, trainable = train)
        relu1 = tf.nn.relu(input_, name = 'enc_relu1_layer{}'.format(layer_no))
        conv1 = ops.conv1d(relu1, options['residual_channels'], name = 'enc_conv1d_1_layer{}'.format(layer_no))
        # conv1 = layer_norm(conv1, trainable = train)
        conv1 = conv1 * source_mask
        relu2 = tf.nn.relu(conv1, name = 'enc_relu2_layer{}'.format(layer_no))
        dilated_conv = ops.conv1d(relu2, options['residual_channels'], 
            dilation, options['encoder_filter_width'],
            causal = True, 
            name = "enc_dilated_conv_layer{}".format(layer_no)
            )
        # dilated_conv = layer_norm(dilated_conv, trainable = train)
        dilated_conv = dilated_conv * source_mask
        relu3 = tf.nn.relu(dilated_conv, name = 'enc_relu3_layer{}'.format(layer_no))
        conv2 = ops.conv1d(relu3, 2 * options['residual_channels'], name = 'enc_conv1d_2_layer{}'.format(layer_no))
        conv2 = conv2 * source_mask
        return input_ + conv2

def main():
    # for testing
    options = {
        'encoder_dilations' : [1,2,4],
        'residual_channels' : 2,
        'encoder_filter_width' : 3
    }

    source_sentence = tf.placeholder('float32', [None, 4, 2], name = "ss")
    print source_sentence
    debug_tensors = encoder_bytenet(source_sentence, options)
    sess = tf.InteractiveSession()
    tf.initialize_all_variables().run()
    sent_1 = [ [1,2], [4,5], [7,8], [0,0] ]
    sent_2 = [ [1,2], [4,5], [0,0], [0,0] ]
    snp = np.array([sent_1] + [sent_2])
    print snp
    m1, m2 = sess.run([debug_tensors['encoder_output'], debug_tensors['last_seq_element']], 
        feed_dict = {
            source_sentence : snp,
        })
    print "******"
    print m1
    print '------'
    print m2

if __name__ == '__main__':
    main()