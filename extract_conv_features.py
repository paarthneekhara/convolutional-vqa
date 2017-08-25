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
    parser.add_argument('--bucket_size', type=int, default=10000,
                       help='Bucket Size')
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
    image_id_list_all = [img_id for img_id in image_ids]
    print "Total Images", len(image_id_list_all)
    print "initialising features array"
    
    # LIMITED RAM, tough life :/
    image_buckets, image_bucket_mapping = create_buckets(image_id_list_all, args.bucket_size)
    print "Made buckets"

    try:
        shutil.rmtree('Data/conv_features_{}'.format(args.split))
    except:
        pass
    os.makedirs('Data/conv_features_{}'.format(args.split))

    with open('Data/bucket_data_{}.p'.format(args.split), 'wb') as f:
        data = {
            'image_buckets' : image_buckets,
            'total_buckets' : len(image_buckets),
            'image_bucket_mapping' : image_bucket_mapping
        }
        pickle.dump(data, f)

    print "Saved bucket data"
    
    if args.model=="vgg":
        cnn_model = vgg16.create_vgg_model(448, only_conv = args.feature_layer != 'fc7')
    else:
        cnn_model = resnet.create_resnet_model(448)

    for bucket_no, image_id_list in enumerate(image_buckets):
        if args.feature_layer == "fc7":
            conv_features = np.ndarray( (len(image_id_list), 4096) , dtype = 'float32')
            img_dim = 224
        else:
            if args.model=="vgg":
                conv_features = np.ndarray( (len(image_id_list), 14, 14, 512) , dtype = 'float32')
                img_dim = 448
            else:
                conv_features = np.ndarray( (len(image_id_list), 14, 14, 2048) , dtype = 'float32')
                img_dim = 448
                print "it's done!!!"

        print "initialised feature array"

        

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
            print "Images Processed", idx, bucket_no

            

        print "Saving conv_features features"

        

        file_name = "Data/conv_features_{}/conv_features_{}_bucket_{}.h5".format(args.split, args.feature_layer, bucket_no)
        h5f_conv_features = h5py.File( file_name, 'w')
        h5f_conv_features.create_dataset('conv_features', data=conv_features)
        h5f_conv_features.close()

        print "Saving image id list"
        file_name = "Data/conv_features_{}/image_id_list_bucket_{}.h5".format(args.split, bucket_no)
        h5f_image_id_list = h5py.File( file_name, 'w')
        h5f_image_id_list.create_dataset('image_id_list', data=image_id_list)
        h5f_image_id_list.close()
        print "Done!", bucket_no, len(image_buckets)

def create_buckets(image_id_list, bucket_size):
    buckets = []
    while len(image_id_list) > bucket_size:
        buckets.append(image_id_list[:bucket_size])
        image_id_list = image_id_list[bucket_size:]
    buckets.append(image_id_list)

    image_bucket_mapping = {}
    for bucket_no,image_id_list in enumerate(buckets):
        for img_id in image_id_list:
            image_bucket_mapping[img_id] = bucket_no

    return buckets, image_bucket_mapping

if __name__ == '__main__':
    main()