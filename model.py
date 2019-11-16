import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import numpy as np
import pandas as pd
# import sklearn
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import tensorflow as tf
import keras
from keras import backend as K
from keras import Input
from keras.models import Model, load_model
from keras.layers import Cropping2D, Conv2D, BatchNormalization
from keras.layers import ReLU, Flatten, Dense, Dropout, Lambda
import sys
import os
import math
import glob
import csv
import moviepy.editor as mpy


root_path = '../'
project_path = root_path + 'CarND-Behavioral-Cloning-P3/'
model_path = project_path + 'model/'
train_data_path = project_path + 'data_train/'
predict_data_path = project_path + 'data_predict/'
record_data_path = project_path + 'data_record/'
image_path = project_path + 'images/'
output_path = project_path + 'output/'


class DataList:
    def __init__(self, zero_frac=0.1, use_side_camera=False):
        self.X_all = None
        self.y_all = None
        self.X_train = None
        self.y_train = None
        self.X_valid = None
        self.y_valid = None
        self.X_test = None
        self.y_test = None
        self.zero_frac = zero_frac
        self.use_side_camera = use_side_camera
        self.original_df = None
        self.augmented_df = None
        
    def load_data(self, folder=train_data_path):
        self.original_df = self.read_csv(folder)
        self.augmented_df = self.data_augmentation()

    def read_csv(self, folder=train_data_path):
        # Remove the space char at the beginning of the image file name 
        def strip(text):
            try:
                return text.strip()
            except AttributeError:
                return text

        def make_float(text):
            return float(text)

        df = pd.read_csv(folder + 'driving_log.csv',
                         sep=',',
                         converters = {'center': strip,
                                       'left': strip,
                                       'right': strip}
                        )
        print('size: {}'.format(df.shape))
        return df
    
    def merge_csv(self, auto_save=True, folder=record_data_path):
        # Parse all subfolders under folder/, except IMG
        folders = []
        for entry in os.scandir(folder):
            if entry.is_dir() and entry.name.find('.') < 0 and entry.name != 'IMG':
                folders.append(folder + entry.name + '/')
        frames = [self.read_csv(folder) for folder in folders]
        df = pd.concat(frames)
        left = df['left'].apply(lambda x: x[x.find('IMG'):].replace('\\', '/'))
        right = df['right'].apply(lambda x: x[x.find('IMG'):].replace('\\', '/'))
        center = df['center'].apply(lambda x: x[x.find('IMG'):].replace('\\', '/'))
        df['left'] = left
        df['right'] = right
        df['center'] = center
        if auto_save:
            df.to_csv(path_or_buf=folder+'driving_log.csv', index=False,)
        print('total df size: {}'.format(df.shape))
        return df

    def data_augmentation(self, show_hist=False):
        df = self.original_df
        
        step = 0.1
        x_label = ['{:>.2f}'.format(x + step / 2) for x in np.arange(-1.0, 1.0, step).tolist()]
        if show_hist:
            # Show original data histogram
            # Group them by column 'steering' with step = 0.1
            groups = df.groupby(pd.cut(df["steering"], np.arange(-1.0, 1.0+step, step), labels=x_label))
            fig = plt.figure(1, figsize=(10, 5))
            rects = plt.bar(x_label, groups['steering'].count(), align="center", width=0.95)
            plt.title("Histogram of original steering angles")
            plt.xlabel("Steering")
            plt.ylabel("Frequency")
            plt.show()
            plt.close()
        
        # Filter panda DataFrame with condition 'steering = 0.0'
        # Only keep 10% of data whose 'steering = 0.0'
        zero_df = df[(df['steering'] < 1e-6) & (df['steering'] > -1e-6)]
        sample_zero_df = zero_df.sample(frac=self.zero_frac, random_state=None)
        not_zero_df = df[(df['steering'] >= 1e-6) | (df['steering'] <= -1e-6)]
        self.augmented_df = pd.concat([sample_zero_df, not_zero_df])
        # print('df shape: {}'.format(df.shape))
        # print('zero_df shape: {}'.format(zero_df.shape))
        # print('not_zero_df shape: {}'.format(not_zero_df.shape))
        # print('sample_zero_df shape: {}'.format(sample_zero_df.shape))
        # print('augmented_df shape: {}'.format(self.augmented_df.shape))
                
        if show_hist:
            # Show augmented data histogram
            # Group them by column 'steering' with step = 0.1
            groups = self.augmented_df.groupby(
                pd.cut(self.augmented_df["steering"],
                       np.arange(-1.0, 1.0+step, step),
                       labels=x_label))
            fig = plt.figure(1, figsize=(10, 5))
            rects = plt.bar(x_label, groups['steering'].count(), align="center", width=0.95)
            plt.title("Histogram of augmented steering angles")
            plt.xlabel("Steering")
            plt.ylabel("Frequency")
            plt.show()
            plt.close()

        if self.use_side_camera:
            # Include the left and right images, and fine tune their steering angles.
            df = self.augmented_df
            X = np.array(df['center'].tolist())
            left_X = np.array(df['left'].tolist())
            right_X = np.array(df['right'].tolist())
            y = np.array(df['steering'].tolist())
            left_y = self.steering_transform(np.array(df['steering'].tolist()), left=True)
            right_y = self.steering_transform(np.array(df['steering'].tolist()), left=False)
            self.X_all = np.concatenate((X, left_X, right_X))
            self.y_all = np.concatenate((y, left_y, right_y))
            X_train, X_test, y_train, y_test = train_test_split(self.X_all, self.y_all, test_size=0.2)
            self.X_train, self.X_test = X_train, X_test
            self.y_train, self.y_test = y_train, y_test
            # X_test, X_valid, y_test, y_valid = train_test_split(X_test, y_test, test_size=0.5)
            # self.X_train, self.X_test, self.X_valid = X_train, X_test, X_valid
            # self.y_train, self.y_test, self.y_valid = y_train, y_test, y_valid
        else:
            df = self.augmented_df
            X = np.array(df['center'].tolist())
            y = np.array(df['steering'].tolist())
            self.X_all = X
            self.y_all = y
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X_all, self.y_all, test_size=0.2)
        print('X_all: {}, y_all: {}'.format(self.X_all.shape, self.y_all.shape))
        print('X_train: {}, y_train: {}'.format(self.X_train.shape, self.y_train.shape))
        # print('X_valid: {}, y_valid: {}'.format(self.X_valid.shape, self.y_valid.shape))
        print('X_test: {}, y_test: {}'.format(self.X_test.shape, self.y_test.shape))

    def steering_transform(self, y, left=True):
        correction = 0.2
        if left == True:
            y += correction
        else:
            y -= correction
        return y

def generator(X, y, batch_size=64, flipped=True):
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
        
    while 1: # Loop forever so the generator never terminates
        X, y, y_flip_tag = shuffle(X, y, y_flip_tag)
        for offset in range(0, data_size, batch_size):
            X_batch = X[offset:offset+batch_size]
            flip_tag_batch = y_flip_tag[offset:offset+batch_size]

            images = []
            for name, flip in zip(X_batch, flip_tag_batch):
                # print('name: {}'.format(name))
                # print('Read image: {}'.format(data_path + name))
                img = mpimg.imread(train_data_path + name)
                if flip == 1:
                    img = np.fliplr(img)
                images.append(img)
                
            # trim image to only see section with road
            X_train = np.array(images)
            y_train = y[offset:offset+batch_size]
            # TODO: Do some image augmentation here.
            # yield sklearn.utils.shuffle(X_train, y_train)
            yield X_train, y_train

class PilotNet(keras.callbacks.Callback):
    def __init__(self):
        self.model = None
        self.salient_model = None
        self.cropping_top = 50
        self.cropping_down = 20
        
    def build_model(self):
        self.build_pilotnet_model()
        self.build_salient_model()
        
    def build_pilotnet_model(self):
        def resize(images):
            import tensorflow as tf
            return tf.image.resize_area(images, size=(66, 200))
        
        def preprocessing(images):
            import tensorflow as tf
            standardization = lambda x: tf.image.per_image_standardization(x)
            normalized_rgb = tf.math.divide(images, 255.0)
            augmented_img_01 = tf.image.random_brightness(normalized_rgb, max_delta=0.4)
            augmented_img_02 = tf.image.random_contrast(augmented_img_01, lower=0.5, upper=1.5)
            std_img = tf.map_fn(standardization, augmented_img_02)
            return std_img
        
        cropping_top, cropping_down = self.cropping_top, self.cropping_down

        # Build PilotNet model
        inputs = keras.Input(shape=(160, 320, 3), name='inputs')
        cropping = keras.layers.Cropping2D(cropping=((cropping_top, cropping_down), (0, 0)),
                                           name='cropping')(inputs)
        resize = keras.layers.Lambda(resize, name='resize')(cropping)
        norm = keras.layers.Lambda(preprocessing, name='normalization')(resize)
        # input = (66, 200, 3), output = (31, 98, 24)
        conv_1 = keras.layers.Conv2D(filters=24,
                                     kernel_size=5,
                                     strides=2,
                                     padding='valid',
                                     name='conv_1')(norm)
        bn_1 = keras.layers.BatchNormalization(name='bn_1')(conv_1)
        relu_1 = keras.layers.ReLU(name='relu_1')(bn_1)
        
        # input = (31, 98, 24), output = (14, 47, 36)
        conv_2 = keras.layers.Conv2D(filters=36,
                                     kernel_size=5,
                                     strides=2,
                                     padding='valid',
                                     name='conv_2')(relu_1)
        bn_2 = keras.layers.BatchNormalization(name='bn_2')(conv_2)
        relu_2 = keras.layers.ReLU(name='relu_2')(bn_2)
        
        # input = (14, 47, 36), output = (5, 22, 48)
        conv_3 = keras.layers.Conv2D(filters=48,
                                     kernel_size=5,
                                     strides=2,
                                     padding='valid',
                                     name='conv_3')(relu_2)
        bn_3 = keras.layers.BatchNormalization(name='bn_3')(conv_3)
        relu_3 = keras.layers.ReLU(name='relu_3')(bn_3)
        
        # input = (5, 22, 48), output = (3, 20, 64)
        conv_4 = keras.layers.Conv2D(filters=64,
                                     kernel_size=3,
                                     strides=1,
                                     padding='valid',
                                     name='conv_4')(relu_3)
        bn_4 = keras.layers.BatchNormalization(name='bn_4')(conv_4)
        relu_4 = keras.layers.ReLU(name='relu_4')(bn_4)
        
        # input = (3, 20, 64), output = (1, 18, 64)
        conv_5 = keras.layers.Conv2D(filters=64,
                                     kernel_size=3,
                                     strides=1,
                                     padding='valid',
                                     name='conv_5')(relu_4)
        bn_5 = keras.layers.BatchNormalization(name='bn_5')(conv_5)
        relu_5 = keras.layers.ReLU(name='relu_5')(bn_5)
        
        # input = (1, 18, 64), output = (1152,)
        flatten = keras.layers.Flatten(name='flatten')(relu_5)
        dropout_5 = keras.layers.Dropout(rate=0.5, name='dropout_5')(flatten)
        
        # input = (1152,), output = (100,)
        dense_6 = keras.layers.Dense(units=100,
                                     name='dense_6')(dropout_5)
        bn_6 = keras.layers.BatchNormalization(name='bn_6')(dense_6)
        relu_6 = keras.layers.ReLU(name='relu_6')(bn_6)
        dropout_6 = keras.layers.Dropout(rate=0.5, name='dropout_6')(relu_6)
        
        # input = (100,), output = (50,)
        dense_7 = keras.layers.Dense(units=50,
                                     name='dense_7')(dropout_6)
        bn_7 = keras.layers.BatchNormalization(name='bn_7')(dense_7)
        relu_7 = keras.layers.ReLU(name='relu_7')(bn_7)
        dropout_7 = keras.layers.Dropout(rate=0.5, name='dropout_7')(relu_7)
        
        # input = (50,), output = (10,)
        dense_8 = keras.layers.Dense(units=10,
                                     name='dense_8')(dropout_7)
        bn_8 = keras.layers.BatchNormalization(name='bn_8')(dense_8)
        relu_8 = keras.layers.ReLU(name='relu_8')(bn_8)
        dropout_8 = keras.layers.Dropout(rate=0.5, name='dropout_8')(relu_8)
        
        # input = (10,), output = (1,)
        outputs = keras.layers.Dense(units=1,
                                     name='outputs')(dropout_8)


        # set up cropping2D layer
        self.model = keras.Model(inputs=inputs, outputs=outputs)
        self.model.compile(optimizer=keras.optimizers.Adam(0.01),
                           loss=keras.losses.logcosh,
                           # loss='mse',
                           metrics=['mse', 'mae'])
        
    def build_salient_model(self):
        def resize_norm(images):
            import tensorflow as tf
            max_val = tf.math.reduce_max(images)
            min_val = tf.math.reduce_min(images)
            norm = tf.math.divide(tf.math.subtract(images, min_val), max_val - min_val)
            resize = tf.image.resize_area(norm, size=(90, 320))
            return resize
        
        # Build salient model
        relu_5 = self.model.get_layer('relu_5').output
        relu_4 = self.model.get_layer('relu_4').output
        relu_3 = self.model.get_layer('relu_3').output
        relu_2 = self.model.get_layer('relu_2').output
        relu_1 = self.model.get_layer('relu_1').output
        
        ave_5 = keras.layers.Lambda(lambda x: K.mean(x, axis=3, keepdims=True), name='ave_5')(relu_5)
        deconv_5 = keras.layers.Conv2DTranspose(filters=1,
                                                kernel_size=3,
                                                strides=1,
                                                padding='valid',
                                                output_padding=(0, 0),
                                                use_bias=False,
                                                kernel_initializer=keras.initializers.Ones(),
                                                name='deconv_5')(ave_5)
        
        ave_4 = keras.layers.Lambda(lambda x: K.mean(x, axis=3, keepdims=True), name='ave_4')(relu_4)
        mul_4 = keras.layers.Multiply(name='mul_4')([ave_4, deconv_5])
        deconv_4 = keras.layers.Conv2DTranspose(filters=1,
                                                kernel_size=3,
                                                strides=1,
                                                padding='valid',
                                                output_padding=(0, 0),
                                                use_bias=False,
                                                kernel_initializer=keras.initializers.Ones(),
                                                name='deconv_4')(mul_4)
        
        ave_3 = keras.layers.Lambda(lambda x: K.mean(x, axis=3, keepdims=True), name='ave_3')(relu_3)
        mul_3 = keras.layers.Multiply(name='mul_3')([ave_3, deconv_4])
        deconv_3 = keras.layers.Conv2DTranspose(filters=1,
                                                kernel_size=5,
                                                strides=2,
                                                padding='valid',
                                                output_padding=(1, 0),
                                                use_bias=False,
                                                kernel_initializer=keras.initializers.Ones(),
                                                name='deconv_3')(mul_3)
        
        ave_2 = keras.layers.Lambda(lambda x: K.mean(x, axis=3, keepdims=True), name='ave_2')(relu_2)
        mul_2 = keras.layers.Multiply(name='mul_2')([ave_2, deconv_3])
        deconv_2 = keras.layers.Conv2DTranspose(filters=1,
                                                kernel_size=5,
                                                strides=2,
                                                padding='valid',
                                                output_padding=(0, 1),
                                                use_bias=False,
                                                kernel_initializer=keras.initializers.Ones(),
                                                name='deconv_2')(mul_2)
        
        ave_1 = keras.layers.Lambda(lambda x: K.mean(x, axis=3, keepdims=True), name='ave_1')(relu_1)
        mul_1 = keras.layers.Multiply(name='mul_1')([ave_1, deconv_2])
        deconv_1 = keras.layers.Conv2DTranspose(filters=1,
                                                kernel_size=5,
                                                strides=2,
                                                padding='valid',
                                                output_padding=(1, 1),
                                                use_bias=False,
                                                kernel_initializer=keras.initializers.Ones(),
                                                name='deconv_1')(mul_1)
        salient_mask = keras.layers.Lambda(resize_norm, name='salient_mask')(deconv_1)
        
        self.salient_model = keras.Model(inputs=self.model.input, outputs=salient_mask)
        self.salient_model.compile(optimizer=keras.optimizers.Adam(0.01),
                                   loss='mse', metrics=['mse'])
        
        return self.salient_model
        
    def predict_generator(self, generator, steps):
        result = self.model.predict_generator(generator=generator, steps=steps)
        return result
    
    def train(self, train_generator, train_steps, epochs, verbose,
              valid_generator, valid_step):
        history = self.model.fit_generator(
            generator=train_generator,
            steps_per_epoch=train_steps,
            epochs=epochs,
            callbacks=[self],
            verbose=verbose,
            validation_data=valid_generator,
            validation_steps=valid_step)
        return history
    
    def on_epoch_end(self, epoch, logs={}):
        if epoch % 5 == 0 and epoch != 0:
            print('on_epoch_end: {}'.format(epoch))
            self.save_model(model_path + "model_{:0>3d}.h5".format(epoch), model_type='weight')
    
    def get_salient_image(self, generator, steps):
        top = self.cropping_top
        salient_images = []
        mask_images = []
        mask_color = [14, 252, 26]
        for i in range(steps):
            images, steers = next(generator)
            mask = self.salient_model.predict(images)
            mask = np.tile(mask, 3)
            mask[:,:,:,0] *= mask_color[0]
            mask[:,:,:,1] *= mask_color[1]
            mask[:,:,:,2] *= mask_color[2]
            full_mask = np.zeros_like(images)
            full_mask[:,top:top+mask.shape[1],:,:] = mask
            for i in range(full_mask.shape[0]):
                input_img = images[i]
                mask_img = full_mask[i]
                salient_img = cv2.addWeighted(input_img, 1.0, mask_img, 0.8, 0)
                salient_images.append(salient_img)
                mask_images.append(mask_img)
        return salient_images, mask_images
    
    def save_model(self, name, model_type='entire'):
        if name.endswith('h5') == False:
            name += '.h5'
        if model_type == 'weight':
            self.model.save_weights(name)
        elif model_type == 'entire':
            self.model.save(name)
        
    def load_model(self, name, model_type='entire'):
        if model_type == 'weight':
            # Restore the model's state,
            # this requires a model with the same architecture.
            if self.model == None:
                self.build_model()
            self.model.load_weights(name)
        elif model_type == 'entire':
            # Recreate the exact same model, including weights and optimizer.
            self.model = keras.models.load_model(name)
            self.build_salient_model()

class SimpleDataList:
    def __init__(self):
        self.X = None
        self.y = None
        self.original_df = None
        
    def load_data(self, folder=predict_data_path):
        self.original_df = self.read_csv(folder)
        self.augmented_df = self.data_augmentation()

    def read_csv(self, folder=predict_data_path):
        # Remove the space char at the beginning of the image file name 
        def strip(text):
            try:
                return text.strip()
            except AttributeError:
                return text

        def make_float(text):
            return float(text)

        df = pd.read_csv(folder + 'driving_log.csv',
                         sep=',',
                         converters = {'center': strip,
                                       'left': strip,
                                       'right': strip}
                        )
        print('size: {}'.format(df.shape))
        return df

    def data_augmentation(self, show_hist=True):
        df = self.original_df
        self.X = np.array(df['center'].tolist())
        self.y = np.array(df['steering'].tolist())
        
def simple_generator(X, y, batch_size=1, flipped=False):
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
        
    while 1: # Loop forever so the generator never terminates
        # X, y, y_flip_tag = shuffle(X, y, y_flip_tag)
        for offset in range(0, data_size, batch_size):
            X_batch = X[offset:offset+batch_size]
            flip_tag_batch = y_flip_tag[offset:offset+batch_size]

            images = []
            for name, flip in zip(X_batch, flip_tag_batch):
                # print('name: {}'.format(name))
                # print('Read image: {}'.format(data_path + name))
                img = mpimg.imread(predict_data_path + name)
                if flip == 1:
                    img = np.fliplr(img)
                images.append(img)
                
            # trim image to only see section with road
            X_train = np.array(images)
            y_train = y[offset:offset+batch_size]
            # TODO: Do some image augmentation here.
            # yield sklearn.utils.shuffle(X_train, y_train)
            yield X_train, y_train

if __name__ == '__main__':
    if len(sys.argv) == 2:
        function = sys.argv[1]
    else:
        function = 'train'
    print('Run function: {}'.format(function))
    
    if function == 'train':
        # Load data csv file, zero_frac = 0.1
        data_list = DataList(zero_frac=0.6)
        data_list.load_data(folder=train_data_path)
        # Set generator batch size
        batch_size = 128
        # compile and train the model using the generator function
        train_generator = generator(data_list.X_train, data_list.y_train, batch_size=batch_size)
        test_generator = generator(data_list.X_test, data_list.y_test, batch_size=batch_size)
        # valid_generator = generator(data_list.X_valid, data_list.y_valid, batch_size=batch_size)
        train_step = math.ceil(data_list.X_train.shape[0] * 2 / batch_size)
        test_step = math.ceil(data_list.X_test.shape[0] * 2 / batch_size)
        # valid_step = math.ceil(data_list.X_valid.shape[0] * 2 / batch_size)
        
        pilot_net = PilotNet()
        pilot_net.build_model()
        epochs = 100
        verbose = 1
        history = pilot_net.train(train_generator, train_step, epochs, verbose, test_generator, test_step)
        weight_name = model_path + 'model_weight.h5'
        model_name = model_path + 'model.h5'
        pilot_net.save_model(model_name, model_type='entire')
        pilot_net.save_model(weight_name, model_type='weight')
        
        plt.plot(history.history['loss'])
        plt.plot(history.history['mean_squared_error'])
        plt.plot(history.history['mean_absolute_error'])
        plt.plot(history.history['val_loss'])
        plt.plot(history.history['val_mean_squared_error'])
        plt.plot(history.history['val_mean_absolute_error'])
        plt.title('model mean squared error loss')
        plt.ylabel('mean squared error loss')
        plt.xlabel('epoch')
        plt.legend(['train loss', 'train mse', 'train mae',
                    'valid loss', 'valid mse', 'valid mae'], loc='upper right')
        plt.savefig(output_path + 'history.png', bbox_inches='tight')
        plt.close()
    elif function == 'salient':
        # Load model
        weight_name = model_path + 'model_weight.h5'
        pilot_net = PilotNet()
        pilot_net.load_model(weight_name, model_type='weight')
        # Prepare data
        simple_data_list = SimpleDataList()
        simple_data_list.load_data(folder=predict_data_path)
        # Create generator
        gif_generator = simple_generator(simple_data_list.X, simple_data_list.y)
        gif_step = simple_data_list.X.shape[0]
        # Prediction
        salient_output, mask_imgs = pilot_net.get_salient_image(gif_generator, gif_step)
        # print('salient_output complete')
        # Output video file
        gif_name = output_path + 'salient_yuv_object'
        fps = 30
        clip = mpy.ImageSequenceClip(salient_output, fps=fps)
        clip.write_videofile(gif_name + ".mp4",fps=fps)
        clip.write_gif('{}.gif'.format(gif_name), fps=fps)
    elif function == 'merge_data':
        data_list = DataList(zero_frac=0.15)
        data_list.merge_csv(auto_save=True, folder=record_data_path)
    elif function == 're_save_model':
        pilot_net = PilotNet()
        weight_name = model_path + 'model_weight.h5'
        model_name = model_path + 'model.h5'
        pilot_net.load_model(weight_name, model_type='weight')
        pilot_net.save_model(model_name, model_type='entire')
    else:
        # Do nothing
        print('Unknown function')
