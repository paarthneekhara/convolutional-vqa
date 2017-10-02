from tensorflow.contrib import slim
from tensorflow.contrib.slim.nets import resnet_v2, resnet_utils
import tensorflow as tf
import cnn_preprocessing
from scipy import misc
import json
import numpy as np
import labels

# download model from http://download.tensorflow.org/models/resnet_v2_152_2017_04_14.tar.gz
# list of other models https://github.com/tensorflow/models/tree/70b894bdf74c4deedafbbdd70c2454162837d5d2/slim
def create_resnet_model(img_dim):
    pre_image = tf.placeholder(tf.float32, [None, None, 3])
    processed_image = cnn_preprocessing.preprocess_for_eval(pre_image/255.0, img_dim, img_dim)

    images = tf.placeholder(tf.float32, [None, img_dim, img_dim, 3])
    # mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
    # processed_images = images - mean
    with slim.arg_scope(resnet_utils.resnet_arg_scope()):
        probs, endpoints = resnet_v2.resnet_v2_152(images, num_classes=1001, is_training = False)
        print endpoints['resnet_v2_152/block4']

    init_fn = slim.assign_from_checkpoint_fn(
            'Data/CNNModels/resnet_v2_152.ckpt',
            slim.get_model_variables('resnet_v2_152'))

    sess = tf.Session()
    init_fn(sess)

    return {
        'images_placeholder' : images,
        'block4' : endpoints['resnet_v2_152/block4'],
        'session' : sess,
        'processed_image' : processed_image,
        'pre_image' : pre_image,
        'probs' : probs
    }

def main():
    def load_image_array(image_file, img_dim):
        img = misc.imread(image_file)
        if len(img.shape) == 2:
            img_new = np.ndarray( (img.shape[0], img.shape[1], 3), dtype = 'float32')
            img_new[:,:,0] = img
            img_new[:,:,1] = img
            img_new[:,:,2] = img
            img = img_new
        if not img_dim:
            return img
        img_resized = misc.imresize(img, (img_dim, img_dim))
        return img_resized
    res = create_resnet_model(448)
    sess = res['session']
    new_image = sess.run(res['processed_image'], feed_dict = {
        res['pre_image'] : load_image_array('0.jpg', img_dim = None)
        })
    # print "1", load_image_array('0.jpg', 224)
    check = sess.run(res['probs'], feed_dict = {
        res['images_placeholder'] : [new_image]
    })
    # print labels.label_names[np.argmax(check)], np.max(check)
    
    
    check = check[0,0,0]
    print "shape", check.shape
    preds = (np.argsort(check)[::-1])[0:5]
    print preds
    for p in preds:
        print labels.label_names[p-1]
        # print labels.label_names[np.argmax(check)], np.max(check)
    print "2", check
    # print "3", check2

if __name__ == '__main__':
    main()