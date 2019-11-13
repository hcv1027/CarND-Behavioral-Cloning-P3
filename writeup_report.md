# **Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[pilot_net]: ./writeup_images/PilotNet.png "Pilot Net"
[pilot_net_model_v1]: ./writeup_images/pilot_net_model_v1.png "Pilot Net Model Version 1"
[pilot_net_model_v2]: ./writeup_images/pilot_net_model_v2.png "Pilot Net Model Version 2"
[salient_net_model]: ./writeup_images/salient_net_model.png "Salient Net Model"
[data_augmented_01]: ./writeup_images/data_augmented_01.png "Flipping image"
[data_augmented_02]: ./writeup_images/data_augmented_02.png "Data Distribution"
[training_history]: ./writeup_images/history.png "Training history"
[histogram_1]: ./writeup_images/histogram_1.png "Steering distribution before resample"
[histogram_2]: ./writeup_images/histogram_2.png "Steering distribution after resample"
[visual_back_prop]: ./writeup_images/VisualBackProp.png "Block diagram of the VisualBackProp method"
[salient_object]: ./writeup_images/salient_object.png "Salient objects found by PilotNet"
[autonomous_mode]: ./writeup_images/autonomous_mode.gif "Autonomous mode"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* [model.py](./model.py) containing the script to create and train the model
* [drive.py](./drive.py) for driving the car in autonomous mode
* [model.h5](./model/model.h5) containing a pre-trained keras model
* [model_weight.h5](./model/model_weight.h5) containing a pre-trained keras model's weight
* [writeup_report.md](./writeup_report.md) or writeup_report.pdf summarizing the results
* [video.mp4](./video.mp4) a video recording when in autonomous mode

#### 2. Submission includes functional code
Using the Udacity provided simulator and my [drive.py](./drive.py) file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

---
### How to Use

#### 1. Dependency
Here is the list of tensorflow and keras version I used to train and save my model.
* tensorflow (1.12.0)
* Keras (2.2.4)

#### 2. Training
Put the training data into the folder `data_train`. Then use the command `python model.py train`, it will train 100 epochs and output two file to the folder `model`, one is the model's weight named as `model_weight.h5`, the other is a keras whole model named as `model.h5`

#### 3. Fix Keras version conflict problem
Since your working environment may be different with mine, I provide a command which can just load my pre-trained model's weight, and then save it to a new keras whole model again in your working environment. This may fix the version conflict problem. You can use the command: `python model.py re_save_model` to achieve this requirement.

---
### Referenced papers

I reference these four papers to implement my model architecture.
1. [End to End Learning for Self-Driving Cars](https://arxiv.org/abs/1604.07316)
2. [Explaining How a Deep Neural Network Trained with End-to-End Learning Steers a Car](https://arxiv.org/abs/1704.07911)
3. [VisualBackProp: efficient visualization of CNNs](https://arxiv.org/abs/1611.05418)
4. [A guide to convolution arithmetic for deep learning](https://arxiv.org/abs/1603.07285)

Here is the architecture of the model which Nvidia named **PilotNet**:
![PilotNet][pilot_net]

---
### Model Architecture and Training Strategy

#### 1. Data Augmentation

I use four methods to augment my training data.
1. **Using left and right camera**: To let my model learn how to recover from the side of the road, I use the images catched from left and right cameras in my training process. I adjust the steering measurement by the value `0.2` and `-0.2` with respective to left and right images. This part is done in `model.py` from #TODO through #TODO.
2. **Flipping images**: I flip the image horizontally, and then take the opposite sign of the steering measurement to prevent my trained model to have left turn bias. The part is done inside my `generator`
   ```python
   def generator(X, y, batch_size=32, flipped=True):
        # Flipping the image to create more training data
        if flipped == True:
            X_flip = np.copy(X)
            y_flip = np.copy(y) * -1
            zero_tag = np.zeros_like(y, dtype=np.int8)
            one_tag = np.ones_like(y, dtype=np.int8)
            y_flip_tag = np.concatenate((zero_tag, one_tag))
            X = np.concatenate((X, X))
            y = np.concatenate((y, y_flip))
        else:
            y_flip_tag = np.zeros_like(y, dtype=np.int8)
        data_size = X.shape[0]
   ```
3. **Prepare more training data**: To let my model be more generalized, I use the simulator to collect more training data. Specifically, I turn the car around and drive counter-clockwise laps around both simple and hard track. I also collect some record data when the car is driving from the side of the road back toward the center line.
4. **Resample data**: Since the most part of record data whose steering angle are zero. In the training process, I only randomly choose 15% of data whose steering measurement are zero. It can prevent my trained model to be tend to predict zero steering angle all the time and also save the training time. This part is done in `model.py` from #TODO through #TODO. Below are the histogram of steering distribution before and after I resample them.
   
   ![histogram_1][histogram_1]
   ![histogram_2][histogram_2]
5. **Image Preprocessing**: To let my model more robust, I use some image adjustment techniques to preprocess the input RGB-space images, and then convert to YUV-space:
   ```python
   def preprocessing(images):
            import tensorflow as tf
            import numpy as np
            
            # return tf.map_fn(lambda img: tf.image.per_image_standardization(img), images)
            random_noise = lambda x: tf.image.random_jpeg_quality(x, 50, 100)
    
            shift_y = np.zeros(shape=(66, 200, 3)).astype(np.float32)
            shift_y[:,:,0] += 0.5
            normalized_rgb = tf.math.divide(images, 255.0)
            # noise_img = tf.image.random_jpeg_quality(normalized_rgb, 50, 100)
            noise_img = tf.map_fn(random_noise, normalized_rgb)
            augmented_img_01 = tf.image.random_brightness(noise_img, max_delta=0.8)
            augmented_img_02 = tf.image.random_contrast(augmented_img_01, lower=0.8, upper=3.0)
            augmented_img_03 = tf.image.random_hue(augmented_img_02, max_delta=0.03)
            augmented_img_04 = tf.image.random_saturation(augmented_img_03,lower=0,upper=4)
            max_val = tf.math.reduce_max(augmented_img_04)
            min_val = tf.math.reduce_min(augmented_img_04)
            norm_augmented = tf.math.divide(tf.math.subtract(augmented_img_04, min_val), max_val - min_val)
            yuv = tf.image.rgb_to_yuv(norm_augmented)
            yuv_norm = tf.subtract(yuv, shift_y)
            return yuv_norm
   ```

#### 2. Training stragegy

![pilot_net_model_v1][pilot_net_model_v1]

Above image shows my first version of model architecture. I just follow the original PilotNet model architecture introduced in the paper *"End to End Learning for Self-Driving Cars"* to implement my model. But this model doesn't work well. It always predict steerning angle as a constant value. And the record of loss values shown in the training process didn't change much. All signs tell me that this model learned nothing in the process of training. So I started to analysis what's going on. I exzamined the output values layer by layer, and immediatelly noticed that the output of dense_3 were always zero. But the kernel weights and bias of dense_3 were not zeros, it did't make sense. After deeper analysis, I found the problem was that the result computing `dot(input, kernel) + bias` were all negative, after passed these result into relu activation function, the outputs all became zeros. I find that the root cause is I do not normalize the output of each layer. Once the outputs of `dot(input, kernel) + bias` are negative values, they become zero after relu activation function, finally always output a constant value (In my case, it is bias of dense_3). And because the most outputs do not change, the loss doesn't change much too, so the gradient descent algorithm can't work well. I guess if I let my model train longer time, it can finally learn something, but it will be very inefficient.

So I add a batch normalization layer between each convolution (or fully-connection layer) and activation layer. I also add a droupout layer with `rate = 0.5` after the activation layer of last three fully-connected layers. After this modification, my model works well to predict steering algles. Below is my final model architecture.

![pilot_net_model_v2][pilot_net_model_v2]

#### 3. Training history

Below diagram shows the traing history output from keras. You can see that the training loss and validation loss decrease steadily. There is no overfitting or underfitting im my model.

![Training history][training_history]

#### 4. Finding the salient object

Since I fill the part of finding the salient objects introduced in the paper *"Explaining How a Deep Neural Network Trained with End-to-End Learning Steers a Car"* is very meaningful, I decide to implement it in my project. The concept of how to visualize what PilotNet learned is shown below:

![VisualBackProp][visual_back_prop]

And here is my model architecture to complete this work:

![SalientNet model][salient_net_model]

I use the formula listed in the Keras document to compute the `output_padding`.
```python
new_rows = ((rows - 1) * strides[0] + kernel_size[0]
            - 2 * padding[0] + output_padding[0])
new_cols = ((cols - 1) * strides[1] + kernel_size[1]
            - 2 * padding[1] + output_padding[1])
```
You can find the code about how to build SalientModel and how to blend the input image with the salient objects found by my model in the below code snipper:
```python
class PilotNet:
    def build_salient_model(self):
        ### How to build the SalientNet model
        ...
        return self.salient_model
    
    def get_salient_image(self, generator, steps):
        ### How to blend the input image with salient object found by PilotNet
        ...
        return salient_images
```

Here are some sample images which mark the salient objects found by my PilotNet model:
![salient_object][salient_object]

#### 5. Test my model through simulator

Here is a short video shows the car controlled by my trained model in simulator.


### Conclusion and future work

In this project, I learned how the batch normalization can help to train the deep neural network. I also get deeper understand about how the convolution and transposed convolution work. My next goal is to let my model can auto drive on the harder road in simulator.

