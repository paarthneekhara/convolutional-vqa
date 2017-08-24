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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=str, default='train',
                       help='train/val/test')
    
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch Size')
    parser.add_argument('--version', type=int, default=2,
                       help='VQA dataset version 1 or 2')
    parser.add_argument('--feature_layer', type=str, default="block4",
                       help='CONV FEATURE LAYER, fc7, pool5 or block4')
    parser.add_argument('--model', type=str, default="resnet",
                       help='CONV FEATURE LAYER')
    


    args = parser.parse_args()
    
    all_data = data_loader.load_data(version = args.version)
    
    if args.split == "train":
        qa_data = all_data['training']
    else:
        qa_data = all_data['validation']
    
    image_ids = {}
    for qa in qa_data:
        image_ids[qa['image_id']] = 1

    image_id_list = [img_id for img_id in image_ids]
    print "Total Images", len(image_id_list)
    

    print "initialising features array"
    if args.feature_layer == "fc7":
        conv_features = np.ndarray( (len(image_id_list), 4096) , dtype = 'float32')
        img_dim = 224
    else:
        if args.model=="vgg":
            conv_features = np.ndarray( (len(image_id_list), 14, 14, 512) , dtype = 'float16')
            img_dim = 448
        else:
            conv_features = np.ndarray( (len(image_id_list), 7, 7, 2048) , dtype = 'float16')
            img_dim = 224
            print "it's done!!!"

    print "initialised feature array"

    if args.model=="vgg":
        cnn_model = vgg16.create_vgg_model(img_dim, only_conv = args.feature_layer != 'fc7')
    else:
        cnn_model = resnet.create_resnet_model(img_dim)

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
            image_batch[i,:,:,:] = utils.load_image_array(image_file, img_dim)
            idx += 1
            count += 1
        
        
        feed_dict  = { images : image_batch[0:count,:,:,:] }
        conv_features_batch = sess.run(image_feature_layer, feed_dict = feed_dict)
        conv_features[(idx - count):idx] = conv_features_batch[0:count]

        end = time.clock()
        print "Time for batch of photos", end - start
        print "Hours Remaining" , ((len(image_id_list) - idx) * 1.0)*(end - start)/60.0/60.0/args.batch_size
        print "Images Processed", idx

        

    print "Saving conv_features features"

    file_name = "Data/conv_features_{}_{}_{}.h5".format(args.version, args.split, args.feature_layer)
    h5f_conv_features = h5py.File( file_name, 'w')
    h5f_conv_features.create_dataset('conv_features', data=conv_features)
    h5f_conv_features.close()

    print "Saving image id list"
    file_name = "Data/image_id_list_{}_{}.h5".format(args.version, args.split)
    h5f_image_id_list = h5py.File( file_name, 'w')
    h5f_image_id_list.create_dataset('image_id_list', data=image_id_list)
    h5f_image_id_list.close()
    print "Done!"

if __name__ == '__main__':
    main()