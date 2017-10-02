import tensorflow as tf
from scipy import misc
from os import listdir
from os.path import isfile, join
import data_loader
import utils
import argparse
import numpy as np
import pickle
import h5py
import time
from Models import vgg16, resnet
import json
import shutil
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=str, default='train',
                       help='train/val/test')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch Size')
    parser.add_argument('--feature_layer', type=str, default="block4",
                       help='CONV FEATURE LAYER, fc7, pool5 or block4')
    parser.add_argument('--model', type=str, default="resnet",
                       help='CONV FEATURE LAYER')
    args = parser.parse_args()
    
    
    
    if args.split == "train":
        with open('Data/annotations/captions_train2014.json') as f:
            images = json.loads(f.read())['images']
    else:
        with open('Data/annotations/captions_val2014.json') as f:
            images = json.loads(f.read())['images']
    
    image_ids = {image['id'] : 1 for image in images}
    image_id_list = [img_id for img_id in image_ids]
    print "Total Images", len(image_id_list)
    
    try:
        shutil.rmtree('Data/conv_features_{}_{}'.format(args.split, args.model))
    except:
        pass
    
    os.makedirs('Data/conv_features_{}_{}'.format(args.split, args.model))
    
    if args.model=="vgg":
        cnn_model = vgg16.create_vgg_model(448, only_conv = args.feature_layer != 'fc7')
    else:
        cnn_model = resnet.create_resnet_model(448)
    
    
    image_id_file_name = "Data/conv_features_{}_{}/image_id_list_{}.h5".format(args.split, args.model, args.feature_layer)
    h5f_image_id_list = h5py.File( image_id_file_name, 'w')
    h5f_image_id_list.create_dataset('image_id_list', data=image_id_list)
    h5f_image_id_list.close()

    conv_file_name = "Data/conv_features_{}_{}/conv_features_{}.h5".format(args.split, args.model, args.feature_layer)
    hdf5_conv_file = h5py.File( conv_file_name, 'w')

    if args.feature_layer == "fc7":
        conv_features = None
        feature_shape =  (len(image_id_list), 4096)
        img_dim = 224
    
    else:
        if args.model=="vgg":
            conv_features = None
            feature_shape =  (len(image_id_list), 14, 14, 512)
            img_dim = 448
        else:
            conv_features = None
            feature_shape =  (len(image_id_list), 14*14*2048)
            img_dim = 448
            print "it's done!!!"

    hdf5_data = hdf5_conv_file.create_dataset('conv_features', feature_shape,
                                            dtype='f')
    
    sess = cnn_model['session']
    images = cnn_model['images_placeholder']
    image_feature_layer = cnn_model[args.feature_layer]

    idx = 0
    while idx < len(image_id_list):
        start = time.clock()

        image_batch = np.ndarray( (args.batch_size, img_dim, img_dim, 3 ) )

        count = 0
        for i in range(0, args.batch_size):
            if idx >= len(image_id_list):
                break

            image_file = join('Data', '%s2014/COCO_%s2014_%.12d.jpg'%(args.split, args.split, image_id_list[idx]) )

            if args.model == 'resnet':
                image_array = sess.run(cnn_model['processed_image'], feed_dict = {
                    cnn_model['pre_image'] : utils.load_image_array(image_file, img_dim = None)
                    })
            else:
                image_array = utils.load_image_array(image_file, img_dim = img_dim)
            
            image_batch[i,:,:,:] = image_array
            idx += 1
            count += 1
        
        
        feed_dict  = { images : image_batch[0:count,:,:,:] }
        conv_features_batch = sess.run(image_feature_layer, feed_dict = feed_dict)
        conv_features_batch = np.reshape(conv_features_batch, ( conv_features_batch.shape[0], -1 ))
        hdf5_data[(idx - count):idx] = conv_features_batch[0:count]

        end = time.clock()
        print "Time for batch of photos", end - start
        print "Hours Remaining" , ((len(image_id_list) - idx) * 1.0)*(end - start)/60.0/60.0/args.batch_size
        print "Images Processed", idx

    hdf5_conv_file.close()
    print "Done!"

if __name__ == '__main__':
    main()