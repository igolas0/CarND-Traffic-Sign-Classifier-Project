
[//]: # (Image References)

[image1]: ./examples/cnn_operations.png "Transfer Learning"
[image2]: ./examples/normalize.png "Normalize"
[image3]: ./examples/datagenerator.png "Generator"
[image4]: ./examples/barchart.jpg "Visualization"
[image5]: ./examples/100kmh_v2.jpg "Traffic Sign 100km/h"
[image6]: ./examples/dataaug1.jpeg "Data Augmented Image 1"
[image7]: ./examples/dataaug2.jpeg "Data Aug 2"
[image8]: ./examples/dataaug3.jpeg "Data Aug 3"
[image9]: ./examples/dataaug4.jpeg "Data Aug 4"
[image10]: ./examples/features.png "Dataset features"
[image11]: ./web_images/roundabout_mandatory.jpg "Web DL image"
[image12]: ./web_images/100kmh.jpg "Web DL image"
[image13]: ./web_images/stop.jpg "Web DL image"
[image14]: ./web_images/priority_road.jpg "Web DL image"
[image15]: ./web_images/double_curve.jpg "Web DL image"
[image16]: ./web_images/go_straight_or_right.jpg "Web DL image"
[image17]: ./web_images/80kmh_2.jpg "Web DL image"
[image18]: ./web_images/60kmh.jpg "Web DL image"
[image19]: ./web_images/30kmh.jpg "Web DL image"
[image20]: ./web_images/80kmh.jpg "Web DL image"
[image21]: ./examples/transfer_learning.png "Transfer Learning2"


## Traffic Sign Recognition using a Convolutional Neural Network

---
### Overview
In this project a convolutional neural network is trained to classify traffic signs. The model is trained and validated so it can classify traffic sign images using the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset). Then the performance of the model is also tested on (german) traffic signs downloaded from the web.

---
### The Project
The project is divided in the following steps:

* Loading the data set
* Exploring and visualizing the data set
* Designing, training and testing the model architecture
* Using the model to make predictions on new images (downloaded from the web)
* Analyzing the softmax probabilities of the new images

In the scope of this project I tried two different approaches and thus two different solutions are included:

* Traffic_Sign_Classifier.ipynb (or .html) 
* Traffic_Sign_Classifier_Transfer_Learning.ipynb (or .html) 

My main solution is contained within the first iPython [Notebook](./Traffic_Sign_Classifier.ipynb), where I started building upon a LeNet architecture and ended up using a modified AlexNet convolutional neural network architecture.

Then I also tried a different approach, which you can find in this other [notebook](./Traffic_Sign_Classifier_Transfer_Learning.ipynb). Mostly for learning purposes I wanted to test the concept of Transfer Learning via using a pre-trained network and fine-tuning part of it on the dataset at hand. Here I ended up opting for the well-known ResNet50 architecture (which is available in Keras with pre-trained weights on the [ImageNet dataset](http://www.image-net.org/)). 

Mainly due to simplicity and convenience I used Keras instead of Tensorflow for this project (Keras with Tensorflow as backend).

### Dataset and Repository

 The dataset was downloaded from this [link](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/5898cd6f_traffic-signs-data/traffic-signs-data.zip), from where a pickle file contains the dataset with already resized images (to 32x32). It contains a training, validation and test set.


#### 1. Data Set Summary & Exploration

I used numpy to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799.
* The size of the validation set is 4410. 
* The size of test set is 12630.
* The shape of a traffic sign image is 32x32x3.
* The number of unique classes/labels in the data set is 43.

#### 2. Visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how much samples the training set contains per traffic sign (ID):

![alt text][image4]

 We can see that for some traffic signs we find as few as around 200 training samples and for others as much as approx. 2000 training samples. My expectation is that the neural net will learn to recognize better the traffic signs with more training samples and viceversa (or in another words the CNN might develop a bias to predicting the classes which have more ocurrences in the training set). 

I will not balance the dataset in order to test the model performance without it, although it is probably a good idea to balance the number of samples per class (e.g. with data augmentation).

Here are some of the images of the dataset which correspond to some of the represented classes:

![alt text][image10]

We can see that the pictures are taken from different angles, lighting conditions, etc. Some of the Traffic Signs are even partly covered by graffitis/painting. The full description of the 43 classes can be founded [here](./signnames.csv).


#### 3. Data preprocessing.

First all training, validation and test sets were shuffled with help of sklearn's library (cell no. 4). Then I also normalized the data to make the input features zero-centered and share the same value range (in this case the standard deviation of the images is computed and all features are divided by the standard deviation. This is important since the optimizer will later have an easier time if the data is zero-centered.  Normalizing the value range (in this case by the standard deviation) also speeds up the optimizing process because it prevents the input feature field of being skewed in some dimensions and also this way some features won't have predeterminately more relevance than others.

Here is a snippet of the code where the normalization takes place (cell no. 5):

![alt text][image2]

As a final preprocessing step we proceed to one-hot encode the labels, that is we transform the original label vector which comes with labels ranging from 1 to 43 into a binary 1xn long vector, where n is 43 in this case (or 42 starting counting from zero). For every image presented to the neural net its job is to output a confidence/probability value which ranges from zero (no confidence at all) to one for each class.

#### 4. Data Augmentation using Keras generator built-in function.

Data augmentation is very often a good idea, especially when we are working with a small dataset. If set up well it also has the potential of helping the neural net to generalize well on to different test-cases or real-life conditions that are not sampled by the data used to train on. It also helps by reducing the number of possible "foolish assumptions or hypothesis" which serve well for the training set but do not apply generally (overfitting). For example if all the training images for one class were taken on a sunny day, the neural net might learn to predict that class for all sunny (test) images, but it actually has learned nothing about the particular features of that class.

Below you will find a snippet of the code describing the image data generator/augmentator:

![alt text][image3]

Random rotations, color channel shifts, random zooms, horizontal and vertical shifts were introduced. I limited the angle of the rotations and did not use vertical flips since that could cause some overlap between some classes, hence confusing the CNN. 

Here are some examples of augmented images:

![alt-text][image6] ![alt-text][image7]![alt-text][image8] ![alt-text][image9]

#### 5. Design and Training of the Model Architecture

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 32x32x32 	|
| RELU					| RELU Activation Function (rectified linear unit)						|
| Max pooling	      	| 2x2 stride,  outputs 16x16x32 				|
| Dropout Layer     	| Dropout layer with 20% probability	|
| Convolution 3x3	    | 1x1 stride, valid padding, outputs 32x32x64 	|
| RELU					| RELU Activation Function (rectified linear unit)						|
| Max pooling	      	| 2x2 stride,  outputs 8x8x64 				|
| Dropout Layer     	| Dropout layer with 20% probability	|
| Convolution 3x3	    | 1x1 stride, valid padding, outputs 8x8x128 |
| RELU					| RELU Activation Function (rectified linear unit)						|
| Max pooling	      	| 2x2 stride,  outputs 4x4x128				|
| Dropout Layer     	| Dropout layer with 20% probability	|
| (FLATTEN)|												|
| Fully connected		| Fully Connected Layer with 2048 neurons		|
| Dropout Layer     	| Dropout layer with 50% probability	|
| Fully connected		| Fully Connected Layer with 2048 neurons		|
| Dropout Layer     	| Dropout layer with 50% probability	|
| Fully connected (output)		| Ouput Layer with 43 neurons/classes		|
| Softmax				| Softmax Activation 	|
|						|												|
|						|												|
 
As you can see in the table above the final architecture consists in three convolutional layers each followed by RELU activations functions, max pooling and a dropout layer. The convolutional layers use 1x1 strides, valid padding and a filter depth of 32, 64 and 128 each. Max Pooling is applied with 2x2 strides and the dropout layers are set to ignore neurons 20% of the time.

On top of these convolutions three fully connected layers with intertwined 50% probability dropout layers are used. The first two FC layers amount 2048 artificial neurons each and the final output layer contains 43 neurons (one per class).

To come up with this architecture I took the well-known AlexNet as reference and modified it making it a bit "smaller" and less complex. The main difference is that my CNN only makes use of three convolutional layers (AlexNet uses five) and extracts less depth out of the convolutional layers. Then I also use half of the neurons in the fully connected layers compared to AlexNet. Reducing the complexity is reasonable since AlexNet is used to classify one thousand different classes on ImageNet and our dataset only contains 43 different classes. The resolution of the input images is lower in our case as well. 

To train the model, I used an Adam optimizer and a batch size of 256. The CNN was trained over 95 epochs on AWS. This architecture achieved 96.1 % accuracy on the test set. I did not record the training and validation accuracies, but they must have been well over 97%. 


The iterative process which led me to the final model is the following:

I started out using the LeNet architecture (with minor changes) and no data augmentation. The LeNet model was quickly overfitting after a few epochs. Introducing some dropout layers did help reduce the overfitting. The dropout layers "silence" by random a predetermined number of neurons in a layer. This forces the network to learn redundant hypothesis to classify the different classes which increases the robustness of the neural network predictions while reducing overfitting. The reason this works is that overfitting happens most of the time when the neural network has a capacity of abstraction which is much higher than the complexity of the input data. So when trained on relative simple datasets (relative to the DNN) the neural network tends to catch on every detail (even on features that do not help generalizing on to new datasets). Hence dropout works because it constraints the neural net to learn only helpful details it can rely on.


 My best result with this combination was 95.8% accuracy on the validation set after only seven epochs, but training over more epochs did not help. The accuracy on the test set was close but lower than 95%.

I also experimented using leaky relu with different alpha value as an activation function instead of relu or using 1x1 convolutions at the beginning of the CNN, but did not observe any improvements in accuracy.

After this I decided that I needed data augmentation to be able to train over more epochs and a bit more complex network to handle a bit higher level of abstraction. This is where I tried different variations of AlexNet and came up with my final model.

The data augmentation did a goob job to reduce overfitting and the CNN was able to continue learning (improving accuracy without overfitting) even after hundred epochs. I got around 97% accuracy on the test set after 300 epochs on an unrecorded run trained on my GTX 1070 graphic card and there were still no signs of overfitting. This means there is a good possibility that accuracy could have been improved further by just training for more epochs (while the learning curve was getting steeper and steeper after 100 epochs). 

Things to try in further work would be other kinds of data augmentation (e.g. zca whitening) and balancing the dataset across the different classes. It seems that the balancing of the classes via data augmentation by itself can improve accuracy by one or several points (percent).

Convolutional Neural Networks work well for computer vision problems since the shared weights help to learn aspect of objects which are independent of the position where they appear in the image. An aspect that still is to be improved is introducing rotation independence to CNN's. This approach is the one to achieve state-of-the-art accuracy (>99% test accuracy) using CNN and [Spatial Transformation](https://papers.nips.cc/paper/5854-spatial-transformer-networks.pdf). The methods used in the paper look very promising and it is something I sure want to play with in the future.
 


#### 6. Test a Model on New Images

Here are ten German traffic signs that I found on the web:

![alt text][image20] ![alt text][image11] ![alt text][image12] ![alt text][image13] ![alt text][image14] 
![alt text][image15] ![alt text][image16] ![alt text][image17] ![alt text][image18] ![alt text][image19] 


Here are the results of the predictions of the model on the downloaded web images, (cell no. 13 in iPython Notebook):

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 80 km/h      		| 80 km/h   									| 
| Roundabout Mandatory | Roundabout Mandatory 										|
| 100 km/h		| 100 km/h								|
| Stop Sign      		| Stop Sign					 				|
| Priority Road			| Priority Road      							|
| Double curve   		| Double Curve	|
| Go straight or right | Go straight or right				|
| 80 km/h	      		| 80 km/h 				|
| 60 km/h	      		| 60 km/h 				|
| 30 km/h	      		| 30 km/h 				|


The model was able to correctly classify 10 of the 10 downloaded traffic signs from the web. This result further validates the accuracy on the test set of 96.1%. 


The code for printing out the top five predictions on the web images is located in the 15th cell of the Ipython notebook.

For the first image, the model is very confident that it is a 80 km/h speed limit sign (probability of 99,9%), and the image does contain a 80 km/h sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .99953      			| 80 km/h   									| 
| .00002     				| 60 km/h										|
| .00002		| 100 km/h								|
| .00001    			| 30 km/h					 				|
| <.00001		    | 50 km/h      							|

Not only the CNN is pretty sure of its correct prediction, but its next guesses would have been similar speed limit signs, so we can be pretty happy with this one.

On seven out of ten images the CNN was very confident (99.9%) about its actually correct predictions. The one where it was least confident about its prediction was the 8th image, which was another 80 km/h speed limit sign. Here the CNN showed 55,5% confidence level. Here is the top 5 probability distribution for this image:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .555      			| 80 km/h   									| 
| .288     				| 50 km/h										|
| .079		| 30 km/h								|
| .030    			| 60 km/h					 				|
| .014		    | 100 km/h      							|

It seems that the first image was a rather ideal (painted) 80 km/h sign and this one was a real picture from the street. Here is the comparison of both (first image on the left):

![alt text][image20] ![alt text][image17]

Still, the CNN was twice as sure about its first guess than about the second one, so its prediction was fairly solid on the 8th image. As we can observe in the table above its next guesses were all speed limit signs.

The other two examples where our model did not show above 99.9% confidence were a 100 km/h (72.1% probability) and a 30 km/h speed limit sign (96.7% probability). It seems that speed limit signs are the most difficult to predict due to the similarity among them.

### 7. Transfer Learning

On a second [iPython Notebook](./Traffic_Sign_Classifier_Transfer_Learning.ipynb) I use Transfer Learning to solve this problem.

I started out experimenting with VGG19 and VGG16, but they were pretty memory intensive even for my GTX 1070 (I was using a generator to resize the images on the fly to 224x224).

Then I got to read this [blog post](http://www.topbots.com/14-design-patterns-improve-convolutional-neural-network-cnn-architecture/) with the awesome chart ploted below:

![alt text][image1] 

This chart helped be judging which architectures offer the best tradeoff between image classification accuracy and computation costs. That is when I decided to try ResNet-50, which offers higher accuracy than VGG for far less computation costs. Inception-v3 could have been a valid choice also.

I made use of the pre-trained weights on ImageNet, which can be loaded directly using [Keras](https://keras.io/applications/#resnet50). Since this model accepts only 224x224 images (or 3 input channels with width and height no smaller than 197 as in the documentation) I decided to resize the images and save them to disk. This would speed up the training process, because it turned out that it was pretty slow having the generator to load and resize the images on the fly. (Note: I am not sure I used openCV's resizing function, which might be more efficient than the ones I used).

The data augmentation method was the same as described above. In this case the Keras Image Generator had to flow the images from directory, as described in the documentation. For this I had to previously organize the pictures and put all pictures which belong to the same class in its own directory as described [here](https://gist.github.com/fchollet/0830affa1f7f19fd47b06d4cf89ed44d).

In Transfer Learning there are several methods to fine tune the pre-trained neural network. E.g.: adapting and training only the last fully connected layers or keeping the weights and fine-tuning some of the intermediate layers or even retraining the whole neural network from scratch.

The image below offers a general idea of where the starting point should be and further steps and tweaking should vary depending on the results:

![alt text][image21] 


The training happens in the sixth cell of the iPython Notebook. My final model loads the pre-trained weights and fine-tunes the conv3_x layers (see description and names of the layers in the ResNet paper) and all the layers on top of that. The weights of the preceding layers are kept fixed during training.

The top of the original ResNet50 (originally average pooling, fc-1000 and softmax) is replaced by two fully connected layers of 4096 neurons each with relu activation functions and by the output layer for the 43 classes plus a softmax activation function.

I got around 98% validation accuracy with this approach after around four epochs of training, as the model was quickly overfitting if trained over more epochs.  

To get the accuracy on the test set I would have to organize all test images into directories by class, which I am skipping for now because it was quite a tedious task and this is not my main solution for the project. I guess with the current model I could have achieved around 97% test accuracy and with some tweaking the results could probably be pushed even further.


