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
from Models import vgg16, resnet_extract

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=str, default='train',
                       help='train/val')
    parser.add_argument('--data_dir', type=str, default='Data',
                       help='Data directory')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch Size')
    parser.add_argument('--version', type=int, default=2,
                       help='Batch Size')
    parser.add_argument('--model_type', type=str, default="vgg",
                       help='VGG or RESNET')
    


    args = parser.parse_args()
    
    all_data = data_loader.load_questions_answers(version = args.version, data_dir=args.data_dir)
    
    if args.split == "train":
        qa_data = all_data['training']
    else:
        qa_data = all_data['validation']
    
    image_ids = {}
    for qa in qa_data:
        image_ids[qa['image_id']] = 1

    image_id_list = [img_id for img_id in image_ids]
    print "Total Images", len(image_id_list)
    
    cnn_model = None
    if args.model_type == "vgg":
        cnn_model = vgg16.create_vgg_model()
    elif args.model_type == "resnet":
        cnn_model = resnet_extract.create_resnet_model()

    sess = cnn_model['session']
    images = cnn_model['images_placeholder']
    image_feature_layer = cnn_model['fc7']

    print "initialising features array"
    conv_features = np.ndarray( (len(image_id_list), 4096) , dtype = 'float32')
    print "initialised feature array"

    idx = 0
    while idx < len(image_id_list):
        start = time.clock()
        image_batch = np.ndarray( (args.batch_size, 224, 224, 3 ) )

        count = 0
        for i in range(0, args.batch_size):
            if idx >= len(image_id_list):
                break
            image_file = join(args.data_dir, '%s2014/COCO_%s2014_%.12d.jpg'%(args.split, args.split, image_id_list[idx]) )
            image_batch[i,:,:,:] = utils.load_image_array(image_file, 224)
            idx += 1
            count += 1
        
        
        feed_dict  = { images : image_batch[0:count,:,:,:] }
        conv_features_batch = sess.run(image_feature_layer, feed_dict = feed_dict)
        conv_features[(idx - count):idx, :] = conv_features_batch[0:count,:]

        end = time.clock()
        print "Time for batch of photos", end - start
        print "Hours Remaining" , ((len(image_id_list) - idx) * 1.0)*(end - start)/60.0/60.0/args.batch_size
        print "Images Processed", idx

        

    print "Saving conv_features features"
    file_name = "conv_features_features_{}_{}_{}.h5".format(args.model_type, args.version, args.split)
    h5f_conv_features = h5py.File( join(args.data_dir, file_name), 'w')
    h5f_conv_features.create_dataset('conv_features_features', data=conv_features)
    h5f_conv_features.close()

    print "Saving image id list"
    file_name = "image_id_list_{}_{}_{}.h5".format(args.model_type, args.version, args.split)
    h5f_image_id_list = h5py.File( join(args.data_dir, file_name), 'w')
    h5f_image_id_list.create_dataset('image_id_list', data=image_id_list)
    h5f_image_id_list.close()
    print "Done!"

if __name__ == '__main__':
    main()