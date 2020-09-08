#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 08:30:11 2020

@author: Billy
"""

import time
import os
import glob
import math
import skimage.measure as measure
from skimage.segmentation import watershed
import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
import random



class CellDetector:
    
    #initialise the Cell Detector
    #This class converts AI outputs into Datasets
    #one must initialise this class with the folder of the AI outputs
    def __init__(self, SaveLoc,outputs = 11, save_prefix = 'predicted'):
        assert type(SaveLoc) == type('')
        
        if not os.path.exists(SaveLoc):
            raise ValueError("Save directory is illegitimate:", SaveLoc,". Please enter the full filepath of an existing directory containing CellSegmentor's outputs.\n")
        else:
            os.chdir(SaveLoc)
            self.net_images = glob.glob("*.png") + glob.glob("*.tif") 
            if len(self.net_images) == 0:
                raise ValueError("Directory", SaveLoc," exists, but there are no Neural Network outputs in this location.")
            self.net_images = [k for k in self.net_images if save_prefix in k]
            if len(self.net_images) == 0:
                raise ValueError("Directory", SaveLoc," exists, but there are no valid images in this directory. Ensure \'", save_prefix,"\' appears in the save name for all Neural Network outputs. This should occur by default; please don't change file names.\n")
        
        print("Found the following outputs:", self.net_images,"\n")
        self.outputs = outputs
        self.save_fold = SaveLoc
        
        
        
    #this function detects all of the cells within a layer of an AI output image and return a dataframe of those cells' properties
    #this function recievees the file location of an image as input.
    #
    #layers: 0 = basal, 1= prickle, 2 = superficial
    def layerDetection(self, imdir, layer = 1, cell_only=True, by_nucleus=False,
                       plot =[], subplot_shape=(0,0), param_open =1, 
                       param_close = 0, param_erosion=3):
        
        assert type(imdir) == type('')
        
        if subplot_shape == (0,0):
            subplot_shape = (int(len(plot)/2)+1, 2)
        elif not(subplot_shape[0]*subplot_shape[1] == len(plot) or subplot_shape[0]*subplot_shape[1] == len(plot)-1):
            print(subplot_shape)
            print(subplot_shape[0]*subplot_shape[1])
            print(len(plot))
            print(subplot_shape[0]*subplot_shape[1] != len(plot) or subplot_shape[0]*subplot_shape[1] != len(plot)-1)
            print("Input Subplot shape is invalid. Automatically being changed to:",(int(len(plot)/2), 2))
            subplot_shape = (int(len(plot)/2)+1, 2)
        
        if not os.path.exists(imdir):
            raise ValueError("Image directory is illegitimate:", imdir,"\n")
            
        dicter = {}
        dicter[0] = 'Basal'
        dicter[1] = 'Prickle'
        dicter[2] = 'Superficial'
        
        if not (layer in dicter):
            raise ValueError("'Layer' arg must be an int between 0-2, corresponding to the layers like such:",dicter)
            
        parent_dir, file = os.path.split(imdir)
        imdata = np.array(Image.open(imdir).convert('L')).astype('uint8')
        
        if layer == 0:
            image = np.where(np.logical_or(imdata==255,imdata==230),255,0).astype('uint8')
            nuclei = np.where(imdata==255,255,0).astype('uint8')
        elif layer ==1:
            image = np.where(np.logical_or(imdata==204,imdata==179),255,0).astype('uint8')
            nuclei = np.where(imdata==204,255,0).astype('uint8')
        else:
            image = np.where(np.logical_or(imdata==153,imdata==128),255,0).astype('uint8')
            nuclei = np.where(imdata==128,255,0).astype('uint8')
    
        kernel = np.ones((3,3),np.uint8)
        opening = cv2.morphologyEx(image,cv2.MORPH_CLOSE,kernel, iterations = param_open)
        erosion = cv2.erode(opening, kernel,iterations = param_erosion)
        closing = cv2.morphologyEx(erosion,cv2.MORPH_OPEN,kernel, iterations = param_close)
        
        if not by_nucleus:
            ret, markers = cv2.connectedComponents(closing)
        else:
            nuclei = cv2.morphologyEx(nuclei,cv2.MORPH_OPEN,kernel, iterations = 1)
            ret, markers = cv2.connectedComponents(nuclei)

        # Marker labelling
        markers = markers.astype(np.int32)
        
        labels = watershed(image, markers, mask=cv2.morphologyEx(image,cv2.MORPH_OPEN,kernel, iterations = 1))
        
        plot_dict = {'AI':imdata,'Watershed':labels,'Original':np.asarray(Image.open(os.path.join(parent_dir,file[10:]))), 'Markers':markers, 'Nuclei':nuclei, 'Binary':image}
        
        if len(plot)==0:
            pass
        elif len(plot) == 1:
            if plot[0] in plot_dict:
                plt.plot(plot_dict[plot[0]])
                plt.title(plot[0])
            else:
                print("Your desired plot",plot[0],"is not available. Chose from this list:",plot_dict.keys())
        else:
            
            index = 0
            while not (plot[index] in plot_dict.keys()):
                index+=1
                
            ax1 = plt.subplot(subplot_shape[0],subplot_shape[1], 1)
            ax1.title.set_text(plot[index])
            ax1.imshow(plot_dict[plot[index]])
                
            for i in range(len(plot)):
                i = i+index
                try:
                    plt.subplot(subplot_shape[0],subplot_shape[1],i+1, sharex=ax1, sharey=ax1)
                    plt.imshow(plot_dict[plot[i]])
                    plt.title(plot[i])
                except:
                    print(plot[i],"is not available to plot, so has been skipped. Only choose values from:",plot_dict.keys())
        
        table = measure.regionprops(labels)
        num_cells = len(table)

        if not cell_only:
            table_nuclei = measure.regionprops(watershed(markers, markers, mask=markers))
            table.extend(table_nuclei)

        lister =[]
        if layer == 0:
            layer='Basal'
        elif layer == 1:
            layer = 'Prickle'
        else:
            layer = 'Superficial'
        
        for i,cell in enumerate(table):
            entry ={}
            if i+1>num_cells:
                entry['Type'] = 'nucleus'
            else:
                entry['Type'] = 'cell'
            circ = (4*np.pi*cell['Area'])/(cell['Perimeter']**2)
            if circ >100:
                circ = np.nan
            area = (cell['convex_area']/2)+cell['perimeter']
            entry['Origin_Image'] = os.path.splitext(file)[0]
            entry['Layer'] = layer
            entry['Identifier'] = cell['label']
            entry['Area'] = area
            if circ == np.nan:
                entry['Perimeter'] = cell['perimeter']*1.1
            else:
                entry['Perimeter'] = math.sqrt((2*area)/circ)
            entry['Centroid_y'] = cell['centroid'][0]
            entry['Centroid_x'] = cell['centroid'][1]
            entry['Solidity'] = cell['solidity']
            entry['Major axis diameter'] = cell['major_axis_length']/2
            entry['Minor axis diameter'] = cell['minor_axis_length']/2
            entry['Circularity'] = circ
            lister.append(entry)
            
        return lister
    
    
    #extracts all of the cell data from each layer in an image and saves a dataframe of that data
    def predictOne(self, imdir = None,isRandom = False,cell_only=True, isOpt=True):
    
        if isOpt: optimise = [(0,3,0),(1,0,1),(1,2,1),(2,0,2),(1,3,2),(4,1,3),(1,3,0)]
        if imdir == None:
            if isRandom:
                img_choice = random.randint(0,len(self.net_images)-1)
            else:
                choice_dict = {}
                for i,file in enumerate(self.net_images):
                    choice_dict[i] = file
                ask_str = "Choose one of the following images:\n"+ str(choice_dict)+ " \n"
                img_choice = int(input(ask_str))
                
            imdir = os.path.join(self.save_fold, self.net_images[img_choice])
        
        data = []
        log = []
        if isOpt:
            for j in range(3):
                data_store = {}
                arr_store = {}
                if j ==0:
                    area_boundary = 100
                else:
                    area_boundary = 250
                for arangement in optimise:
                    opening,erosion,closing = arangement
                    datum = cD.layerDetection(imdir, layer=j, cell_only=cell_only, by_nucleus=False,
                                              param_open=opening, param_close = closing, param_erosion = erosion)
                    key = len(list(filter(lambda d: d['Area'] > area_boundary, datum)))
                    
                    while key in arr_store.keys():
                        key = key+0.01
                    data_store[key] =datum
                    arr_store[key] = str(arangement)+" False"
                    
                    if j ==0:
                        datum = cD.layerDetection(imdir, layer=j, cell_only=cell_only, by_nucleus=True,
                                                  param_open=opening, param_close = closing, param_erosion = erosion)
                        key = len(list(filter(lambda d: d['Area'] > area_boundary, datum)))
                        while key in arr_store.keys():
                            key = key+0.01
                        data_store[key] = datum
                        arr_store[key] = str(arangement)+" True"
                key = np.max(list(arr_store.keys()))
                log.append("Layer "+str(j)+", arrangment: "+arr_store[key])
                data.extend(data_store[key])
        else:
            for i in range(len(self.net_images)): 
                imdir = os.path.join(self.save_fold, self.net_images[i])
                for j in range(3):
                    data.extend(cD.layerDetection(imdir, layer=j, cell_only=cell_only, by_nucleus=False))
            
        df = pd.DataFrame(data)
        save = os.path.join(self.save_fold, "nc_measurements_"+os.path.splitext(os.path.split(imdir)[1])[0]+'.xlsx')
        df.to_excel(save)
        return df
    
    
    #extracts all of the cell data from each layer in each image in a folder and saves a dataframe of that data
    #this function recieves the file location of a folder of images as input
    def predictAll(self, save_loc=None,cell_only=True, isOpt=False):
        #The typical most-succesful configurations of image-tuning parameters
        if isOpt: optimise = [(0,3,0),(1,0,1),(1,2,1),(2,0,2),(1,3,2),(4,1,3),(1,3,0)]
            
        if save_loc == None:
            save_loc = self.save_fold
            
        if not os.path.exists(save_loc):
            raise ValueError("Save directory is illegitimate:", save_loc,". Please enter the full filepath of an existing directory containing CellSegmentor's outputs.\n")
        log = []
        if isOpt:
            for i in range(len(self.net_images)): 
                data = []
                imdir = os.path.join(self.save_fold, self.net_images[i])
                print("Scraping cell data from:",imdir)
                for j in range(3):
                    data_store = {}
                    arr_store = {}
                    if j ==0:
                        area_boundary = 70
                    else:
                        area_boundary = 250
                    for arangement in optimise:
                        opening,erosion,closing = arangement
                        datum = cD.layerDetection(imdir, layer=j, cell_only=cell_only, by_nucleus=False,
                                                  param_open=opening, param_close = closing, param_erosion = erosion)
                        key = len(list(filter(lambda d: d['Area'] > area_boundary, datum)))
                        
                        while key in arr_store.keys():
                            key = key+0.01
                        data_store[key] =datum
                        arr_store[key] = str(arangement)+" False"
                        
                        if j ==0:
                            datum = cD.layerDetection(imdir, layer=j, cell_only=cell_only, by_nucleus=True,
                                                      param_open=opening, param_close = closing, param_erosion = erosion)
                            key = len(list(filter(lambda d: d['Area'] > area_boundary, datum)))
                            while key in arr_store.keys():
                                key = key+0.01
                            data_store[key] = datum
                            arr_store[key] = str(arangement)+" True"
                    key = np.max(list(arr_store.keys()))
                    log.append("Layer "+str(j)+", arrangment: "+arr_store[key])
                    data.extend(data_store[key])
                    
                df = pd.DataFrame(data)
                save = os.path.join(save_loc, "nc_measurements_"+os.path.splitext(self.net_images[i])[0]+'.xlsx')
                df.to_excel(save)
                print("Successfully scraped data. Saving to... ",save)
        else:
            for i in range(len(self.net_images)): 
                data = []
                imdir = os.path.join(self.save_fold, self.net_images[i])
                for j in range(3):
                    data.extend(cD.layerDetection(imdir, layer=j, cell_only=cell_only, by_nucleus=False))
                df = pd.DataFrame(data)
                save = os.path.join(save_loc, "nc_measurements_"+os.path.splitext(self.net_images[i])[0]+'.xlsx')
                df.to_excel(save)
                print("Successfully scraped data. Saving to... ",save)
        print(log)
        return df
    
    
    #aims to clean data, by destroying anaomalous results in a given dataset. This function recieves the file location of the dataset as input.
    # anomalous results are defined by having highly atypical cell area, perimeter and circularity characterisitics.
    def dataCleaner(self,save_loc = None,filename=None,cell_only=True, basal_circularity_bounds=(0.5,1.05), basal_area_bounds = (40,350),
                    prickle_circularity_bounds = (0.5,1.05),prickle_area_bounds=(60,1400), superficial_circularity_bounds=(0.5,1.05),
                    superficial_area_bounds=(60,1500), solidity_bound = 0.7):
        
            
        if save_loc == None:
            save_loc = self.save_fold
            
        if not os.path.exists(save_loc):
            raise ValueError("Save directory is illegitimate:", save_loc,". Please enter the full filepath of an existing directory containing CellSegmentor's outputs.\n")
        
        if filename != None:
            open_ = os.path.join(save_loc, filename)
            non_cleaned = [open_]
        else:
            os.chdir(save_loc)
            non_cleaned = glob.glob("*xlsx*")
            non_cleaned = [k for k in non_cleaned if 'nc_measurement' in k]
            
        
        print(non_cleaned)
        layers = ['Basal','Prickle','Superficial']
        cyto_bounds =[[basal_circularity_bounds,basal_area_bounds],
                       [prickle_circularity_bounds, prickle_area_bounds],
                       [superficial_circularity_bounds, superficial_area_bounds]]
        
        nuc_bounds = [[basal_circularity_bounds,(basal_area_bounds[0]/1.5,basal_area_bounds[1])],
                       [prickle_circularity_bounds, (prickle_area_bounds[0]/1.5, prickle_area_bounds[1]/1.5)],
                       [superficial_circularity_bounds, (superficial_area_bounds[0]*1.5,superficial_area_bounds[1]/1.5) ]]
        
        for image in non_cleaned:
            non_clean =  os.path.join(save_loc,image)
            df = pd.read_excel( non_clean,index=False)
            save_name = os.path.join(save_loc,"clean_data_"+os.path.splitext(image)[0].replace("nc_measurements_","")+'.xlsx')
            cell_store = {'Basal':0, 'Prickle':0, 'Superficial':0}
            for i,layer in enumerate(layers):
                cells = df[(df['Layer']==layer) & (df['Type'] == 'cell')]
                cells.reset_index(drop=True, inplace=True)
                cells = cells[(cells['Circularity']>cyto_bounds[i][0][0]) & (cells['Circularity']<cyto_bounds[i][0][1])]
                cells = cells[(cells['Area']>cyto_bounds[i][1][0]) & (cells['Area']<cyto_bounds[i][1][1])]
                cells = cells[(cells['Circularity']>cyto_bounds[i][0][0]*1.1) & (cells['Area']<cyto_bounds[i][1][1]*0.9)]
                cells = cells[(cells['Circularity']<cyto_bounds[i][0][1]*0.9) & (cells['Area']>cyto_bounds[i][1][0]*1.1)]
                cells = cells[((cells['Solidity']>solidity_bound) | (cells['Area']> 2*cyto_bounds[i][1][0]))]
                cell_store[layer] = cells
                
            if not cell_only:
                nuclei_store = {'Basal':0, 'Prickle':0, 'Superficial':0}
                for i,layer in enumerate(layers):
                    nuclei = df[(df['Layer']==layer) & (df['Type'] == 'nucleus')]
                    nuclei.reset_index(drop=True, inplace=True) 
                    nuclei = nuclei[(nuclei['Circularity']>nuc_bounds[i][0][0]) & (nuclei['Circularity']<nuc_bounds[i][0][1])]
                    nuclei = nuclei[(nuclei['Area']>nuc_bounds[i][1][0]) & (nuclei['Area']<nuc_bounds[i][1][1])]
                    nuclei = nuclei[(nuclei['Circularity']>nuc_bounds[i][0][0]*1.1) & (nuclei['Area']<nuc_bounds[i][1][1]*0.9)]
                    nuclei = nuclei[(nuclei['Circularity']<nuc_bounds[i][0][1]*0.9) & (nuclei['Area']>nuc_bounds[i][1][0]*1.1)]
                    nuclei = nuclei[((nuclei['Solidity']>solidity_bound) | (nuclei['Area']> 2*cyto_bounds[i][1][0]))]
                    nuclei_store[layer] = nuclei
                frames = list(cell_store.values())+list(nuclei_store.values())
            else:
                frames = list(cell_store.values())
            output_df = pd.concat(frames)
            output_df.to_excel(save_name)
            os.remove(non_clean)
    
        
if __name__ == '__main__':
    output_loc = "C:\\Users\\Billy\\Downloads\\Prepared_SVS"
    picLoc = os.path.join(output_loc, 'predicted_cropped_Normal buccal mucosa_0_mag20.png')
    
    cD = CellDetector(output_loc)
    time.sleep(2)
    
    cD.predictOne(picLoc)

    cD.predictAll(output_loc, isOpt=True, cell_only=False)
    
    cD.layerDetection(picLoc,layer=0, plot=['AI','Watershed','Original','Binary'],
                      by_nucleus=False, cell_only=True,subplot_shape=(2,2),
                      param_open=0,  param_erosion = 3, param_close = 0)
    
    data = cD.dataCleaner(output_loc, cell_only=False)

    
    
        