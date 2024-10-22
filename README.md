# feel.io
An incredible AI to detect emotions from the different sources

This is a project for a deep learning course. The decided topic is an app that detects emotions from different sources, such as sound, photos, or videos.  Model can use either one or more sources for detection. 
The result of this project would be a model with a user app that will use this model.


**Project Type**

There are some implementations of such AI as “Affectiva's emotion API” and “DeepEmotion”, which can detect emotions from multiple sources, but this field has not been fully explored yet. 
Although, there was a traditional way to detect emotions, like the Gabor filter[1] with the classification, after the CNN[2] implementation, which was more efficient I cannot call it a classical method nowadays. Therefore, I can classify my project either as “Bring your own method” or “Beat the classics”. Despite this, it is worth noting that I will still consider the main goal as “Bring your own method”, but also will compare it to the “traditional” one in the end.

**Datasets**

Several datasets will be used for training, as multiple sources will be used. List is not completed yet, as I’m still thinking about ways to combine several different datasets. 

**Developing process**

To develop such an app multimodal approach will be utilized, which will combine facial expression recognitions with sound ones and give an output depending on those factors.
The approximate realization time is 3-4 months. \
**Detailed project plan:**\
  •	Datasets analysis – finding the most combinable datasets and combining them: 2 weeks\
  •	Data preprocessing – Cleaning and normalizing datasets: 2-3 weeks\
  •	Model design and build\
    1.	Design and build image processing model: 2-3 weeks \
    2.	Design and build sound processing model: 1-2 weeks\
    3.	Apply transfer learning: 1 week\
    4.	Optimization and checking for possible problems: 1 week\
  •	Training on multiple datasets: 3 weeks\
  •	Fine-tunning the network: Optimising model: 2 weeks\
  •	App development: 1 week\
  •	Writing final report: 2 weeks\
  •	Preparing presentation: 1 week\


**References **
[1] https://www.sciencedirect.com/science/article/abs/pii/S0045790621001890
[2] a cnn-lstm based deep neural networks for facial emotion detection in videos - Arnold Sachith A Hans

