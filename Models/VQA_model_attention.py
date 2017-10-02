import tensorflow as tf
import ops

class VQA_model:
    def __init__(self, options):
        self.options = options

        source_embedding_channels = options['residual_channels']
        self.w_question_embedding = tf.get_variable('w_question_embedding', 
                [options['question_vocab_size'], source_embedding_channels],
                initializer=tf.truncated_normal_initializer(stddev=0.02))

    def build_model(self):
        options = self.options
        
        self.question = tf.placeholder('int32', [None, None], name = "question")
        self.image_features = tf.placeholder('float32', 
            [None, options['img_dim'], options['img_dim'], options['img_channels']], 
            name = "image_features")
        self.answers = tf.placeholder('int32', [None, options['num_answers']], name = "answer")

        # image_features = self.image_features
        image_features = tf.nn.l2_normalize(self.image_features, dim = 3)

        encoded_question = self.encode_question(self.question, options['text_model'], train = True)
        context, prob1, prob2 = self.attend_image(image_features, encoded_question, options['dropout_keep_prob'])

        with tf.variable_scope("post_attention_fc"):
            # context = tf.nn.dropout(context, 0.8)
            # context = tf.nn.tanh(context)
            fc_1 = tf.nn.relu(ops.fully_connected(context, 1024, name = "fc_1"))
            fc_1 = tf.nn.dropout(fc_1, options['dropout_keep_prob'])
            logits = ops.fully_connected(fc_1, options['ans_vocab_size'], name = "logits")

            loss = 0
            for i in range(options['num_answers']):
                loss += tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels = self.answers[:,i], logits = logits)
            loss /= options['num_answers']

            self.loss = tf.reduce_mean(loss)
            self.predictions = tf.argmax(logits,1)

    def build_generator(self, reuse = False):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        options = self.options
        self.g_question = tf.placeholder('int32', [None, None], name = "question")
        self.g_image_features = tf.placeholder('float32', 
            [None, options['img_dim'], options['img_dim'], options['img_channels']], 
            name = "image_features")
        # image_features = self.g_image_features
        image_features = tf.nn.l2_normalize(self.g_image_features, dim = 3)

        encoded_question = self.encode_question(self.g_question, options['text_model'], train = False)
        context, self.g_prob1, self.g_prob2 = self.attend_image(image_features, encoded_question, dropout_keep_prob = 1.0)

        with tf.variable_scope("post_attention_fc"):
            # context = tf.nn.tanh(context)
            fc_1 = tf.nn.relu(ops.fully_connected(context, 1024, name = "fc_1"))
            logits = ops.fully_connected(fc_1, options['ans_vocab_size'], name = "logits")
            self.g_predictions = tf.argmax(logits,1)



    def attend_image(self, image_features, encoded_question, dropout_keep_prob = 1.0):
        options = self.options
        img_dim = options['img_dim']
        img_channels = options['img_channels']
        with tf.variable_scope("Attention"):
            encoded_question_exp = tf.expand_dims(encoded_question, dim = 1)
            encoded_question_tiled = tf.tile(encoded_question_exp, [1, img_dim * img_dim, 1])
            image_features_flat = tf.reshape(image_features, [-1, img_dim * img_dim, img_channels])

            combined_features = tf.concat([image_features_flat, encoded_question_tiled], axis = 2)
            combined_features = tf.nn.dropout(combined_features, dropout_keep_prob)
            combined_features = ops.conv1d(combined_features, 512, name = "conv_1")
            combined_features = tf.nn.relu(combined_features)
            combined_features = tf.nn.dropout(combined_features, dropout_keep_prob)
            logits = ops.conv1d(combined_features, 2, name = "conv_2")
            prob1 = tf.nn.softmax(logits[:,:,0], name = "prob_map_1")
            prob2 = tf.nn.softmax(logits[:,:,1], name = "prob_map_2")

            glimplse1 = tf.reduce_sum(image_features_flat * tf.expand_dims(prob1, dim = 2), 1)
            glimplse2 = tf.reduce_sum(image_features_flat * tf.expand_dims(prob2, dim = 2), 1)

            attention_map1 = tf.reshape(prob1, [-1, img_dim, img_dim])
            attention_map2 = tf.reshape(prob2, [-1, img_dim, img_dim])

            context = tf.concat( [glimplse1, glimplse2, encoded_question], axis = 1, name = "context")
            print context
            print prob1
            print prob2
            return context, attention_map1, attention_map2




    def encode_question(self, question, model_type = "bytenet", train = True):
        options = self.options
        question_mask = tf.sign(question)
        length = tf.cast( tf.reduce_sum(question_mask, 1), tf.int32)

        question_embedding = tf.nn.embedding_lookup(self.w_question_embedding, 
            question, name = "question_embedding")
        question_embedding = tf.nn.tanh(question_embedding)

        dropout_keep_prob = 1
        if train:
            dropout_keep_prob = options['dropout_keep_prob']
        
        # question_embedding = tf.nn.dropout(question_embedding, dropout_keep_prob)
        
        if model_type == "bytenet":
            curr_input = tf.nn.relu( ops.conv1d(
                question_embedding, 2 * options['residual_channels'], 
                name = "question_scale_conv") )
            for layer_no, dilation in enumerate(options['dilations']):
                curr_input = ops.byetenet_residual_block(curr_input, dilation, 
                    layer_no, options['residual_channels'], 
                    options['filter_width'], causal = True, train = train)
            model_output = curr_input
            model_output = ops.last_seq_element(model_output, length)

        elif model_type == "lstm":
            cell = tf.nn.rnn_cell.LSTMCell(num_units=options['residual_channels'], state_is_tuple=True)
            cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob = dropout_keep_prob)
            # cell = tf.nn.rnn_cell.MultiRNNCell([cell] * 2, state_is_tuple=True)
            outputs, last_states = tf.nn.dynamic_rnn(
                cell=cell,
                dtype=tf.float32,
                sequence_length=length,
                inputs=question_embedding)
            model_output = ops.last_seq_element(outputs, length)
        
        return model_output

def main():
    model_options = {
        'question_vocab_size' : 1000,
        'residual_channels' : 512,
        'filter_width' : 3,
        'dilations' : [1, 2, 4, 8, 16,
                          1, 2, 4, 8, 16],
        
        'ans_vocab_size' : 1000,
        'img_dim' : 14,
        'img_channels' : 2048,
        'text_model' : 'lstm',
        'dropout_keep_prob' : 0.8
    }
    model = VQA_model(model_options)
    model.build_model()
    model.build_generator(reuse = True)
if __name__ == '__main__':
    main()
