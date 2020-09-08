# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 16:11:54 2020

@author: Billy
"""
import tensorflow as tf
from PIL import Image
import numpy as np
import math 
import matplotlib.pyplot as plt
import gc
import os
import time
import random
Image.MAX_IMAGE_PIXELS = 933120000

class uNet_segmentor:
    
    #Generates UNet Model using the model weights (supply the file location in the initialising line)
    def __init__(self, checkpoint_loc, first_conv = 8, window = 224, outputs=11):
        
        self.n = n = first_conv
        self.input_size = input_size = (window,window,3)
        self.outputs = outputs
        inputs = tf.keras.layers.Input(input_size)
        conv1 = tf.keras.layers.Conv2D(n, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
        conv1 = tf.keras.layers.Conv2D(n, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
        pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)
        
        conv2 = tf.keras.layers.Conv2D(n*2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
        conv2 = tf.keras.layers.Conv2D(n*2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
        pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)
        
        conv3 = tf.keras.layers.Conv2D(n*(2**2), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
        conv3 = tf.keras.layers.Conv2D(n*(2**2), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
        drop3 = tf.keras.layers.Dropout(0.5)(conv3)
        pool3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(drop3)
        
        conv4 = tf.keras.layers.Conv2D(n*(2**3), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
        conv4 = tf.keras.layers.Conv2D(n*(2**3), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
        drop4 = tf.keras.layers.Dropout(0.5)(conv4)
        pool4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(drop4)
        
        conv_e = tf.keras.layers.Conv2D(n*(2**4), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
        conv_e = tf.keras.layers.Conv2D(n*(2**4), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv_e)
        drop_e = tf.keras.layers.Dropout(0.5)(conv_e)
        pool_e = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(drop_e)
        
        conv5 = tf.keras.layers.Conv2D(n*(2**5), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool_e)
        conv5 = tf.keras.layers.Conv2D(n*(2**5), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
        drop5 = tf.keras.layers.Dropout(0.5)(conv5)
        
        up6_e = tf.keras.layers.Conv2D(n*(2**4), 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(tf.keras.layers.UpSampling2D(size = (2,2))(drop5))
        merge6_e = tf.keras.layers.concatenate([drop_e,up6_e], axis = 3)
        conv6_e = tf.keras.layers.Conv2D(n*(2**4), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6_e)
        conv6_e = tf.keras.layers.Conv2D(n*(2**4), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6_e)
        
        up6 = tf.keras.layers.Conv2D(n*(2**3), 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(tf.keras.layers.UpSampling2D(size = (2,2))(conv6_e))
        merge6 = tf.keras.layers.concatenate([drop4,up6], axis = 3)
        conv6 = tf.keras.layers.Conv2D(n*(2**3), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
        conv6 = tf.keras.layers.Conv2D(n*(2**3), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)
        
        up7 = tf.keras.layers.Conv2D(n*(2**2), 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(tf.keras.layers.UpSampling2D(size = (2,2))(conv6))
        merge7 = tf.keras.layers.concatenate([conv3,up7], axis = 3)
        conv7 = tf.keras.layers.Conv2D(n*(2**2), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
        conv7 = tf.keras.layers.Conv2D(n*(2**2), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)
        
        up8 = tf.keras.layers.Conv2D(n*2, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(tf.keras.layers.UpSampling2D(size = (2,2))(conv7))
        merge8 = tf.keras.layers.concatenate([conv2,up8], axis = 3)
        conv8 = tf.keras.layers.Conv2D(n*2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
        conv8 = tf.keras.layers.Conv2D(n*2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)
            
        up9 = tf.keras.layers.Conv2D(n*2, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(tf.keras.layers.UpSampling2D(size = (2,2))(conv8))
        merge9 = tf.keras.layers.concatenate([conv1,up9], axis = 3)
        conv9 = tf.keras.layers.Conv2D(n*2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
        conv9 = tf.keras.layers.Conv2D(n*2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
        conv9 = tf.keras.layers.Conv2D(self.outputs, 3, activation = 'sigmoid', padding = 'same', kernel_initializer = 'he_normal')(conv9)
        
        model = tf.keras.Model(inputs=inputs, outputs=conv9)
        
        model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001), loss = 'binary_crossentropy', metrics = ['accuracy'])
        
        #If checkpoint path exists, this section loads the checkpoint, updating the model to the latest trained set of weights
        parentdir,file = os.path.split(checkpoint_loc)
        cpkt_save = os.path.join(parentdir,'checkpoint')
        if os.path.exists(cpkt_save):
            model.load_weights(checkpoint_loc)
        
        self.model = model
        
        
        
        
    #Converts an input image into a set of tiles, according to the Neural Networks input size.
    #
    #For example, if the neural network input had an input space of 4x4x3 (4x4 RGB), and the input image to the tile function was of size 7x7x3,
    #the tile function would insert the 7x7x3 input image into an 8x8x3 template (because 8x8 is the nearest set of dimension that is divisible by 4x4 tiles)
    #and then divide that modified template (the 8x8x3 section with the insert image) into 4 4x4x3 sections. 
    #
    #This is useful, because those 4 4x4x3 sections can then be fed directly into the neural network
    def tile(self, im_loc):
        height,width,channels = self.input_size
        
        im = np.asarray(Image.open(im_loc))[:,:,0:3]
        newy = math.ceil(im.shape[0]/self.input_size[0])
        newx = math.ceil(im.shape[1]/self.input_size[1])
        

        template = np.ones((newy*height,newx*width,channels))*255
        template[0:im.shape[0],0:im.shape[1]] = im

        template = template.reshape(newy, height,newx, width,3).swapaxes(1,2).reshape(-1,height,width,3)

        return template, im.shape[0], im.shape[1]
    
    
    
    #predict method receives an input image file location, it opens that image and then allows the AI to analyse it,
    #before returning the AI output.
    #
    #predict method uses the tile function to split the input image into a set of analysable windows, before analysing
    #them, stitching the outputs of the AI together and saving them.
    #
    #the default save location for this function is in te same folder as the input image, where the output file name is identical to the input,
    # except for it being prefixed by 'predicted_'
    def predict(self, im_loc, save_loc=None, show = False):
        
        if save_loc==None:
            parent_dir, file = os.path.split(im_loc)
            save_loc = os.path.join(parent_dir,'predicted_'+file)
        tiles, og_width, og_height = self.tile(im_loc)
        print(np.shape(tiles))
        pred_tiles = self.model.predict(np.array(tiles),verbose=1)
        print(np.shape(pred_tiles))
        
        splitx = math.ceil(og_width/self.input_size[0])
        splity = math.ceil(og_height/self.input_size[1])
        
        pred_tiles = pred_tiles.reshape(splitx,splity,self.input_size[0],self.input_size[1],self.outputs).swapaxes(1,2).reshape(splitx*self.input_size[0],splity*self.input_size[1],self.outputs)
        pred_tiles = np.argmax(pred_tiles,axis=2)
        if show:
            plt.imshow(pred_tiles,cmap='gray', vmax=self.outputs-1, vmin=0).write_png(save_loc)
        else:
            plt.imshow(pred_tiles,cmap='gray', vmax=self.outputs-1, vmin=0).write_png(save_loc)
            plt.close()
        
        
        
    # This method is designed to train the Neural network based on a set of MATLAB app images, but this is not necessary.
    #
    # The MATLAB app is called 'Image Labeller'
    #
    # The MATLAB app allows one to assign semantic labels to an RGB/Greyscale image, by annotating pixel regions with certain colours. Each colour
    # corresponds to a number, which corresponds to a pixel label. Therefore, for each input image, one can generate a 'ground truth' pixel map, where 
    # every pixel is labelled according to the semantic labels set up in the MATLAB App.
    #
    # In reality, this app is not needed for this function. Just have two folders of images; one with with a set of RGB input images and another
    # with a set of ground truths, where the pixel's semantic labels correspond to distinct integers, starting at 0 and incrementing.
    # I.e. In the ground truth, 0 = background, 1 = background, 2 = Stroma etc... The ground truth and corresponding image should be of identical
    # size and shape. Training images and their corresponding ground truths do not need to be the same size as the Neural network input.
    #
    # This function receives the folder lcoations of the training images and ground truths, and chooses random windows to learn from within each corresponding set of images.
    # one can vary how many windows are processed per batch (batch = 32 by default), the number of photos per training image (photos_per_mask = 100 by default) and how many times each
    # training image is revisted (epochs = 500 by default)
    def learn_from_matlab(self,image_loc, label_loc,photo_per_mask=100,batch = 32, epochs=500,colour=True):

        checkpoint_save = "N:\\"+str(self.n)+"_"+str(self.input_size)+"\\cp.ckpt"
        
        assert os.path.exists(image_loc)
        assert os.path.exists(label_loc)
        
        
        def gen(photo_per_mask = photo_per_mask): 
            image_files = os.listdir(image_loc)
            label_files = os.listdir(label_loc)
            gc.collect()
            
            assert len(label_files)== len(image_files)
            for i in range(len(image_files)):
                in_ = np.asarray(Image.open(os.path.join(image_loc,image_files[i])).convert('RGB'))
                label = np.asarray(Image.open(os.path.join(label_loc,label_files[i])).convert('L'))
                
                if colour:
                    in_ = np.asarray(Image.open(os.path.join(image_loc,image_files[i])).convert('RGB'))
                    label = np.asarray(Image.open(os.path.join(label_loc,label_files[i])).convert('L'))
                    padded_image_data = [0]*3
                    for j in range(3):
                        padded_image_data[j] = np.pad(in_[:,:,j], (int(self.input_size[0]/2),int(self.input_size[1]/2)), 'symmetric')
                    padded_image_data = np.dstack((padded_image_data[0],padded_image_data[1],padded_image_data[2]))
                    padded_label_data = np.pad(label, (int(self.input_size[0]/2),int(self.input_size[1]/2)), 'symmetric')
                else:
                    in_ = np.asarray(Image.open(os.path.join(image_loc,image_files[i])).convert('L'))
                    label = np.asarray(Image.open(os.path.join(label_loc,label_files[i])).convert('L'))
                    padded_image_data = np.pad(in_,(int(self.input_size[0]/2),int(self.input_size[1]/2)), 'symmetric')
                    padded_label_data = np.pad(label, (int(self.input_size[0]/2),int(self.input_size[1]/2)), 'symmetric')
                
                for j in range(photo_per_mask):
                    
                    height, width = label.shape
                    row = random.randint(0,height-1)
                    col = random.randint(0,width-1)
                    
                    rotate = random.randint(0,2)
                    
                    window = np.rot90(padded_image_data[row:row+self.input_size[0],col:col+self.input_size[1]],rotate)
                    mask = np.asarray(np.rot90(padded_label_data[row:row+self.input_size[0],col:col+self.input_size[1]],rotate))
                    
                    if random.uniform(0,1)>0.5:
                        window = np.flip(window,axis=0)
                        mask = np.flip(mask, axis=0)
                    
                    yield (window, (np.arange(self.outputs) == mask[...,None]).astype(int))
        
        shuffle = photo_per_mask*len(os.listdir(image_loc))
        dataset = tf.data.Dataset.from_generator(gen, (tf.int64, tf.int64), (self.input_size, (self.input_size[0],self.input_size[1],self.outputs)))
        dataset = dataset.shuffle(shuffle).batch(batch)
        
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save,
                                                         save_weights_only=True,
                                                         verbose=1)
        self.model.fit(dataset,epochs=epochs,callbacks=[cp_callback])

        
if __name__ == '__main__':
    #initialise the unet with the file location of the model checkpoint
    unet = uNet_segmentor('N:\\8_(384, 384, 3)\\cp.ckpt',window = 384, first_conv=8)
    
    #insert the location of input images and files here
    image_loc = "N:\\Bill_Mcgough\\Correct Labelling\\Labels\\ImageData"
    label_loc = "N:\\Bill_Mcgough\\Correct Labelling\\Labels\\MatlabLabelData"
    
    #unet training here based on location of images and files
    unet.learn_from_matlab(image_loc, label_loc, photo_per_mask=65, batch = 10, epochs = 5000, colour = True)
    
    predictions = "C:\\Users\\Billy\\Downloads\\Prepared_SVS"
    images = os.listdir(predictions)
    print(images)
    
    for image in images:
        
        complete_loc = os.path.join(predictions,image)
        print(complete_loc)
        unet.predict(complete_loc)
    
    before = time.time()
    unet.predict("C:\\Users\\Billy\\Downloads\\Output tester\\14-2302 (1)_0_mag20.png")
    end = time.time()
    print(end-before)
