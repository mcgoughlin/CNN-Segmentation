# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 15:54:42 2020

@author: Billy
"""
import CellDetector
import DataPreparation
import unet
import os

#Example Script using all of the created classes:

#assemble all svs into a folder and save the name of the folder as svs_loc
svs_loc = "C:\\Users\\Billy\\Downloads\\Data"

#save the model checkpoint folder, so that it can be loaded when the unet is initialised
model_loc = 'N:\\8_(384, 384, 3)\\cp.ckpt'

#initialise your data prep object with the svs_loc
cd = DataPreparation.DataPreparation(svs_loc)

#initialise your unet model with the location of your .cpkt file, 
#which will be on the same level as all of the wieghts and model parameters
unet = unet.uNet_segmentor(model_loc,window = 384, first_conv=8)

#this method automatically crops and covnerts all svs into png images
cd.AutocropAll()

#this will create subfolders, with converted, cropped version of the svs files.
#These subfolders will contain png files corresponding to images from each svs file.
#loop through these subfolders and analyse them, like so:

#firstly the pngs were saved in a sibling folder to svs_loc.
#this process saves the locations of the pngs as png_loc:
dir_ = os.path.split(svs_loc)[0]
png_loc = os.path.join(dir_, "Prepared_SVS")

#using this location, create a list of subfolders:
subfolders = [x[0] for x in os.walk(png_loc)][1:]



# The next part makes predictions for every image that was generated by cd.AutocropAll(),
#and generates a dataset for each subfolder.
#
#It is useful at this stage to go into each subfolder and check the cropping is correct on all photos. Ideally,
#the png images would be cropped such that only the epithelium is contained. Using your operating system's
#photo manipulation app, rotate/straighten the photo and crop, to cut as much stroma and background out as possible.
for subfolder in subfolders:
    subfolder_loc = os.path.join(dir_,subfolder)
    for file in os.listdir(subfolder):
        file_loc = os.path.join(subfolder_loc,file)
        unet.predict(file_loc)
        
    cd = CellDetector.CellDetector(subfolder_loc, cell_only = True, isOpt = True)
    cd.predictAll(subfolder_loc, isOpt=True, cell_only=False)
    data = cd.dataCleaner(subfolder_loc, cell_only=True)
    
    
    
# if you wish to train the model, you need to provide the location of the training images 
# and their corresponding ground truths, like so:

image_loc = "N:\\Bill_Mcgough\\Correct Labelling\\Labels\\ImageData"
label_loc = "N:\\Bill_Mcgough\\Correct Labelling\\Labels\\MatlabLabelData"
unet.learn_from_matlab(image_loc, label_loc, photo_per_mask=65, batch = 10, epochs = 5000, colour = True)
