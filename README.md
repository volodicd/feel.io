# feel.io
An incredible AI to detect emotions from the different sources

# Project Description

This is a project for a deep learning course. The decided topic is an app that detects emotions from different sources, such as sound, photos, or videos. The model will use either one or more sources for detection.

The result of this project would be a model with a user app that will use this model.

# Project Type

There are some implementations of such AI as “Affectiva's emotion API” and “DeepEmotion”, which can detect emotions from multiple sources, but this field has not been fully explored yet.

Although, there was a traditional way to detect emotions, like the Gabor filter[1] with the classification, after the CNN[2] implementation, which was more efficient, I cannot call it a classical method nowadays. Therefore, I can classify my project either as “Bring your own method” or “Beat the classics”. Despite this, it is worth noting that I will still consider the main goal as “Bring your own method”, but also will compare it to the “traditional” one in the end.

# Datasets

Several datasets will be used for training, as multiple sources will be used. The list is not completed yet, as I’m still thinking about ways to combine several different datasets.

Most probably, the following datasets will be used for image processing:
- **FER2013**
- **Google Facial Expression Comparison Dataset**
- **AffectNet** (if accessible)

Almost all of these datasets combine photos with emotion descriptions. Some of them already have normalized numbered representations of images, but since several datasets will be combined, normalization will be necessary.

For sound processing, the **RAVDESS** dataset will be used. This dataset contains sound files, and the name of the file contains data about the corresponding emotion.

The datasets section may be updated as datasets could change during the processing phase.

# Developing Process

To develop such an app, a multimodal approach will be utilized, which will combine facial expression recognitions with sound ones and give an output depending on those factors.

The approximate realization time is 3-4 months.

## Detailed Project Plan:

- **Datasets analysis** – Finding the most combinable datasets and combining them: 2 weeks.(🚧)
- **Data preprocessing** – Cleaning and normalizing datasets: 2-3 weeks.
- **Model design and build** 
  - Design and build image processing model: 2-3 weeks 
  - Design and build sound processing model: 1-2 weeks
- **Apply transfer learning**: 1 week
- **Optimization and checking for possible problems**: 1 week
- **Training on multiple datasets**: 3 weeks
- **Fine-tuning the network**: Optimizing the model: 2 weeks 
- **App development**: 1 week
- **Writing final report**: 2 weeks
- **Preparing presentation**: 1 week\
(🚧) - we are here
# References 

1. [Gabor filter](https://www.sciencedirect.com/science/article/abs/pii/S0045790621001890)
2. A CNN-LSTM Based Deep Neural Networks for Facial Emotion Detection in Videos - Arnold Sachith A Hans


