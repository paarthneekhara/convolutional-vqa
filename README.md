# Fully Convolutional Visual Question Answering
This is an attention based model for VQA using a dilated convolutional neural network for modelling the question and a resnet for visual features. The text model is based on the recent convolutional architecture ByteNet. Stacked attention distributions over the images are then used to compute weighted image features, which are concatenated with the text features to predict the answer. Following is the rough diagram for the described model.

![Model architecture](http://i.imgur.com/IE6Zq6o.jpg)

## Requirements
- Python 2.7.6
- Tensorflow 1.3.0
- nltk

### Datasets and Paths
- The model is can be trained either on VQA 1.0 or VQA 2.0. Download the dataset by running ```sh download.sh``` in ```Data``` directory.
Unzip the downloaded files and create the directory ```Data/CNNModels```. Download the pretrained Resnet-152 from [here][1] to ```Data/CNNModels```.
- Make 2 empty directories ```Data/Models1```, ```Data/Models2``` for saving the checkpoints during training for VQA 1.0 and 2.0 respectively.

## Training
#### Extract the Image features
- Extract the image features as per the following
  - DEFAULT - Resnet (14,14,2048) block4 features(attention model) - ```python extract_conv_features.py --feature_layer="block4"```
  - VGG (7,7,512) pool5 features(attention model) -  ```python extract_conv_features.py --feature_layer="pool5"```
  - VGG fc7 features (4096,) - ```python extract_conv_features.py --feature_layer="fc7"```

#### Training the attention model
- Train using ```python train_attention.py

[1]:http://download.tensorflow.org/models/resnet_v2_152_2017_04_14.tar.gz
