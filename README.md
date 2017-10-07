# Fully Convolutional Visual Question Answering
This is an attention based model for VQA using a dilated convolutional neural network for modelling the question and a resnet for visual features. The text model is based on the recent convolutional architecture ByteNet. Stacked attention distributions over the images are then used to compute weighted image features, which are concatenated with the text features to predict the answer. Following is the rough diagram for the described model.

![Model architecture](https://i.imgur.com/HZhC2DE.jpg)

## Requirements
- Python 2.7.6
- Tensorflow 1.3.0
- nltk

### Datasets and Paths
- The model is can be trained either on VQA 1.0 or VQA 2.0. Download the dataset by running ```sh download.sh``` in ```Data``` directory.
Unzip the downloaded files and create the directory ```Data/CNNModels```. Download the pretrained Resnet-152 from [here][1] to ```Data/CNNModels```.
- Make 2 empty directories ```Data/Models1```, ```Data/Models2``` for saving the checkpoints while training VQA 1.0 and 2.0 respectively.

## Usage
#### Extract the Image features
- Extract the image features as per the following
  - DEFAULT - Resnet (14,14,2048) block4 features(attention model) - ```python extract_conv_features.py --feature_layer="block4"```
  - VGG (14,14,512) pool5 features(attention model) -  ```python extract_conv_features.py --feature_layer="pool5"```
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
- Use ```python train_evaluate.py --evaluate_every=1 --max_steps=1 --resume_model="Trained Model Path (Data/Models<vqa_-version>/model<epoch>.ckpt)"``` to evaluate a checkpoint.

## Generating Answers/Attention Distributions
#### Pretrained Model
You may download the pretrained model from [here][6]. Save the files in ```Data/Models1```.
- Use ```python generate.py --question="<QUESTION ABOUT THE IMAGE>" --image_file="<IMAGE FILE PATH>" --model_path="<PATH_TO_CHECKPOINT = Data/Models1/model10.ckpt>"``` to generate answer/attention distributions in ```Data/gen_samples```.

## Sample Results
| Image        | Question           | Attention1 |Attention2 | Predicted Answer  |
| ------------- |:-------------:|:-------------:|:-------------:| :-----:|
| ![](https://i.imgur.com/NRxINaq.jpg)|is she going to eat both pizza      | ![](https://i.imgur.com/rxy84Gv.jpg) |![](https://i.imgur.com/fAkQ0VM.jpg) | No |
| ![](https://i.imgur.com/s2jPi0k.jpg)|What color is the traffic light      | ![](https://i.imgur.com/zArjRK0.jpg) |![](https://i.imgur.com/n0qbZst.jpg) | green |
| ![](https://i.imgur.com/ItXZHfK.jpg)|is the persons hair short      | ![](https://i.imgur.com/Upi4VBW.jpg) |![](https://i.imgur.com/xGUurls.jpg) | Yes |
| ![](https://i.imgur.com/LzYcgoS.jpg)|what musical instrument is beside the laptop      | ![](https://i.imgur.com/sjUUi9O.jpg) |![](https://i.imgur.com/QGHtVfk.jpg) | keyboard |
| ![](https://i.imgur.com/wnVqAmd.jpg)|what color hat is the boy wearing      | ![](https://i.imgur.com/yRYlZRe.jpg) |![](https://i.imgur.com/AvtYPvt.jpg) | blue |
| ![](https://i.imgur.com/LLDepgb.jpg)|what are the men doing      | ![](https://i.imgur.com/lr1sw1Y.jpg) |![](https://i.imgur.com/lr1sw1Y.jpg) | eating |
| ![](https://i.imgur.com/ekjP8Yn.jpg)|what type of drink is in the glass      | ![](https://i.imgur.com/gdf5nDE.jpg) |![](https://i.imgur.com/XcjLfZ9.jpg) | orange juice |
| ![](https://i.imgur.com/ZKudDyn.jpg)|is there a house      | ![](https://i.imgur.com/XTiYk7l.jpg) |![](https://i.imgur.com/NX3lV5W.jpg) | yes |


## References
- [Show, Ask, Attend, and Answer: A Strong Baseline For Visual Question Answering][2]
- [Neural Machine Translation in Linear Time][3]
- [Stacked Attention Networks for Image Question Answering][4]
- And other papers/codes from [https://github.com/JamesChuanggg/awesome-vqa][5]

[1]:http://download.tensorflow.org/models/resnet_v2_152_2017_04_14.tar.gz
[2]:https://arxiv.org/abs/1704.03162
[3]:https://arxiv.org/abs/1610.10099
[4]:https://arxiv.org/abs/1511.02274
[5]:https://github.com/JamesChuanggg/awesome-vqa
[6]:https://drive.google.com/file/d/0BzIYiFQpwNAUTmJXY3JYckJhVmM/view?usp=sharing
