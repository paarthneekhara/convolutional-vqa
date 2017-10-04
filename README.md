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

## Usage
#### Extract the Image features
- Extract the image features as per the following
  - DEFAULT - Resnet (14,14,2048) block4 features(attention model) - ```python extract_conv_features.py --feature_layer="block4"```
  - VGG (7,7,512) pool5 features(attention model) -  ```python extract_conv_features.py --feature_layer="pool5"```
  - VGG fc7 features (4096,) - ```python extract_conv_features.py --feature_layer="fc7"```

#### Preprocess Questions/Answers
- Tokeinze the questions/answers using ```python data_loader.py --version=VQA_VERSION``` (1 or 2)

### Training the attention model
- Train using ```python train_evaluate.py --version=VQA_VERSION```
- Following are the customizable model options
  - residual_channels : Number channels in the residual block of bytenet/state of the lstm. Default 512.
  - batch_size : Default 64.
  - learning_rate : default 0.001
  - epochs : Default 25
  - version : VQA dataset version 1 or 2
  - sample_every : sample attention distributions/answers every x steps. Default 200.
  - evaluate_every : Evaluate over validation set every x steps. Default 6000.
  - resume_model : Resume training the model from a checkpoint file.
  - training_log_file : Log accuracy/steps in this filepath. Default 'Data/training_log.json' .
  - feature_layer : Which conv features to use. Default block4 of resnet.
  - text_model : Text model to use : LSTM or bytenet. Default is bytenet
  
### Evaluating a trained model
- The accuracy on the validation is logged every ```evaluate_every``` steps while training the model in ```Data/training_log.json```.
- Use python train_evaluate.py --evaluate_every=1 --max_steps=1 --resume_model="Trained Model Path" to evaluate a checkpoint.

## Generating Answers/Attention Distributions
- Use ```python generate.py --question="<QUESTION ABOUT THE IMAGE>" --image_file="<IMAGE FILE PATH>" --model_path=<"PATH_TO_CHECKPOINT">

## Sample Results
| Image        | Attention1           | Attention2 | Predicted Answer  |
| ------------- |:-------------:|:-------------:| :-----:|
| ![](http://i.imgur.com/j4FiEaS.jpg)      | ![](http://i.imgur.com/j4FiEaS.jpg) |![](http://i.imgur.com/j4FiEaS.jpg) | red, green, yellow|

## References

### Papers



[1]:http://download.tensorflow.org/models/resnet_v2_152_2017_04_14.tar.gz
