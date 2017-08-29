from tensorflow.contrib import slim
from tensorflow.contrib.slim.nets import resnet_v2, resnet_utils
import tensorflow as tf
import cnn_preprocessing
from scipy import misc

def create_resnet_model(img_dim):
    images = tf.placeholder(tf.float32, [None, img_dim, img_dim, 3])
    # resize_size = int((1.0 * img_dim)/224.0 * 256)
    resize_size = img_dim
    # mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
    # check = images-mean
    preprocess_map_fn = lambda x : cnn_preprocessing.preprocess_for_eval( x, img_dim, img_dim, resize_size)
    processed_images = tf.map_fn( preprocess_map_fn, images )
    # processed_images = tf.stack(processed_images_list)
    
    with slim.arg_scope(resnet_utils.resnet_arg_scope()):
        probs, endpoints = resnet_v2.resnet_v2_152(processed_images, num_classes=1001)
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
        'check_images' : processed_images
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

        img_resized = misc.imresize(img, (img_dim, img_dim))
        return img_resized
    res = create_resnet_model(224)
    sess = res['session']
    print "1", load_image_array('0.jpg', 224)
    check = sess.run(res['check_images'], feed_dict = {
        res['images_placeholder'] : [load_image_array('0.jpg', 224)]
        })
    check2 = sess.run(res['check_images'], feed_dict = {
        res['images_placeholder'] : [load_image_array('0.jpg', 224)]
        })
    print "2", check
    print "3", check2

if __name__ == '__main__':
    main()