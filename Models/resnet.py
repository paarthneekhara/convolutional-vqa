from tensorflow.contrib import slim
from tensorflow.contrib.slim.nets import resnet_v2, resnet_utils
import tensorflow as tf

def create_resnet_model(img_dim):
    images = tf.placeholder(tf.float32, [None, 224, 224, 3])
    mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
    images = images-mean
    
    print "imgs", images
    with slim.arg_scope(resnet_utils.resnet_arg_scope()):
        probs, endpoints = resnet_v2.resnet_v2_152(images, num_classes=1001)
        print endpoints['resnet_v2_152/block4']

    init_fn = slim.assign_from_checkpoint_fn(
            'Data/CNNModels/resnet_v2_152.ckpt',
            slim.get_model_variables('resnet_v2_152'))

    sess = tf.Session()
    init_fn(sess)

    return {
        'images_placeholder' : images,
        'block4' : endpoints['resnet_v2_152/block4'],
        'session' : sess
    }

def main():
    create_resnet_model(224)

if __name__ == '__main__':
    main()