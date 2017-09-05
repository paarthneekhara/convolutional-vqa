import tensorflow as tf
import numpy as np
import ops

img_dim = 2
img_channels = 3
image_features = tf.placeholder('float32', 
            [None, img_dim,img_dim,3], 
            name = "image_features")
encoded_question = tf.placeholder('float32', [None, 4])

encoded_question_exp = tf.expand_dims(encoded_question, dim = 1)
encoded_question_tiled = tf.tile(encoded_question_exp, [1, img_dim * img_dim, 1])

image_features_flat = tf.reshape(image_features, [-1, img_dim * img_dim, img_channels])

combined_features = tf.concat([image_features_flat, encoded_question_tiled], axis = 2)
logits = ops.conv1d(combined_features, 2, name = "conv_2")

prob1 = tf.nn.softmax(logits[:,:,0], name = "prob_map_1")
prob2 = tf.nn.softmax(logits[:,:,1], name = "prob_map_2")

glimplse1 = tf.reduce_sum(image_features_flat * tf.expand_dims(prob1, dim = 2), 1)
glimplse2 = tf.reduce_sum(image_features_flat * tf.expand_dims(prob2, dim = 2), 1)

sess = tf.InteractiveSession()
tf.initialize_all_variables().run()

r_image_features = np.random.rand(2,img_dim,img_dim,3)
r_encoded_question = np.random.rand(2,4)
print "image features", r_image_features[1]
print "encoded question", r_encoded_question[1]

test_tensors = [encoded_question_tiled, image_features_flat, combined_features, logits, logits[:,:,1], prob2, glimplse2 ]

results = sess.run(test_tensors, feed_dict = {
	image_features : r_image_features,
	encoded_question : r_encoded_question
})


for r in results:
	print "*********"
	print r[1]
	print "*********"