import numpy as np
from scipy import misc
import tensorflow as tf
from os.path import isfile, join

import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
from skimage import transform, filters

# VGG 16 accepts RGB channel 0 to 1 (This tensorflow model).
# def load_image_array(image_file):
#     img = misc.imread(image_file)
#     # GRAYSCALE
#     if len(img.shape) == 2:
#         img_new = np.ndarray( (img.shape[0], img.shape[1], 3), dtype = 'float32')
#         img_new[:,:,0] = img
#         img_new[:,:,1] = img
#         img_new[:,:,2] = img
#         img = img_new

#     img_resized = misc.imresize(img, (224, 224))
#     return (img_resized/255.0).astype('float32')

def load_image_array(image_file, img_dim = None):
    img = misc.imread(image_file)
    if len(img.shape) == 2:
        img_new = np.ndarray( (img.shape[0], img.shape[1], 3), dtype = 'float32')
        img_new[:,:,0] = img
        img_new[:,:,1] = img
        img_new[:,:,2] = img
        img = img_new

    if img_dim == None:
        return img

    img_resized = misc.imresize(img, (img_dim, img_dim))
    return img_resized

def get_blend_map(img, att_map, blur=True, overlap=True):
    # att_map -= att_map.min()
    # if att_map.max() > 0:
    #     att_map /= att_map.max()
    att_map = 1.0 - att_map
    att_map = transform.resize(att_map, (img.shape[:2]), order = 3, mode='edge')
    # print att_map.shape
    if blur:
        att_map = filters.gaussian(att_map, 0.02*max(img.shape[:2]))
        att_map -= att_map.min()
        att_map /= att_map.max()
    cmap = plt.get_cmap('jet')
    att_map_v = cmap(att_map)
    att_map_v = np.delete(att_map_v, 3, 2)
    if overlap:
        att_map = 1*(1-att_map**0.7).reshape(att_map.shape + (1,))*img + (att_map**0.7).reshape(att_map.shape+(1,)) * att_map_v
    return att_map

def question_indices_to_text(question_indices, ques_vocab_rev):
    question = [ques_vocab_rev[qi] for qi in question_indices]
    return " ".join(question)

def answer_indices_to_text(answer_indices, ans_vocab_rev):
    answers = [ans_vocab_rev[ai] for ai in answer_indices]
    return answers

def image_array_from_image_id(image_id, split):
    image_file = join('Data', '%s2014/COCO_%s2014_%.12d.jpg'%(split, split, image_id) )
    return load_image_array( image_file, 224)

def main():
    img = load_image_array('0.jpg', 224)
    print img.dtype

if __name__ == '__main__':
    main()