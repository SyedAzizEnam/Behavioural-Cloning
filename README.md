# CarND-Project3

# Preprocessing
* Data was cropped to include only the road ignoring other parts of the background while also being transform from RGB to YUV. The image is also resized to be the same dimensions as the images in the Nvidia end-to-end system(https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf).

# Network Architecture
* Experiments were conducted to choose the best architecture for this problem. Initially I had used my model from project 2 as a baseline afterwards I tired different transfer learning approaches by using GoogLeNet/Alexnet. In the end the architecture that worked best on my validation set and on the track was the architecture used in the end-to-end system developed by Nvidia. The network consist of static normalization layer -> 5 convolutional layers -> 4 full connected layers illustrated below:

![alt tag](https://github.com/SyedAzizEnam/CarND-Project3/blob/master/Screen%20Shot%202017-01-17%20at%2010.46.18%20PM.png)

# Training
