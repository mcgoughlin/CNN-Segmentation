# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 13:14:31 2020

@author: Billy
"""

import slideio
import glob
import os
import time
import numpy as np
import gc
import progressbar
from PIL import Image
from skimage import filters


class DataPreparation:
    
    #this function initialises a data cleaner
    #the job of this class is to generate x20 magnification .png images from svs slides,
    #whilst cutting out as much background and non-epithelial area as possible.
    #
    #this function needs the location of the folder that contains svs images as an input
    def __init__(self, svs_loc, png_loc = None):
        assert type(svs_loc) == type('')
        
    
        if not os.path.exists(svs_loc):
            raise ValueError("SVS directory is illegitimate:", svs_loc,". Please enter the full filepath of an existing directory containing .svs files.\n")
        else:
            os.chdir(svs_loc)
            self.svs_images = glob.glob("*.svs")
            if len(self.svs_images) == 0:
                raise ValueError("Directory", svs_loc," exists, but there are no .svs images in this location.")
            self.svs_loc = svs_loc

        print("Found the following outputs:", self.svs_images,"\n")
        
        if png_loc == None:
            parent_dir, dir_ = os.path.split(svs_loc)
            png_loc = os.path.join(parent_dir,"Prepared_SVS")
        
        if not os.path.exists(png_loc):
            os.mkdir(png_loc)
            
        self.png_fold = png_loc


    #this function receives the file location of a folder that contains svs images,
    #and generates a subling subfolder populated with png images.
    #
    #this function also saves a log file, to inform the user about the new image's geometric properties.
    def AutocropAll(self, svs_loc = None, png_loc= None, max_mag= 20):
        
            
        if svs_loc == None:
            svs_loc = self.svs_loc
        if png_loc == None:
            png_loc = self.png_fold
            
        if not os.path.exists(svs_loc):
            raise ValueError("SVS directory is illegitimate:", svs_loc,". Please enter the full filepath of an existing directory containing .svs files.\n")

        if not os.path.exists(png_loc):
            print("File Location", png_loc,"does not exist. Making this directory...")
            os.mkdir(png_loc)
            print("Successfully create .png save directory.")
            

        
        def consecutive(data, stepsize=1):
            arr_consec= np.split(data, np.where(np.diff(data) != stepsize)[0]+1)
            return max(arr_consec, key = len) 

        widgets = [
                'Cropping: ', progressbar.Percentage(),
                ' ', progressbar.AnimatedMarker(),
                ' ', progressbar.ETA(),
            ]


        bar = progressbar.ProgressBar(
        widgets=widgets,
        maxval=len(self.svs_images)).start()
        log_loc = os.path.join(png_loc, 'log.txt')
        
        with open(log_loc, 'w') as filetowrite:
            for i in range(len(self.svs_images)): 
                bar.update(i)
                information ={}
                
                file,svs = os.path.splitext(self.svs_images[i])
                pic1_loc = os.path.join(svs_loc, self.svs_images[i])
                
                
                slide= slideio.open_slide(pic1_loc, 'SVS')
                scene = slide.get_scene(0)
                mag = scene.magnification
                pixel_size = scene.resolution[0]
                _,_,width,height = scene.rect
                
                img_fold = os.path.join(png_loc, file)
                if not os.path.exists(img_fold):
                    os.makedirs(img_fold)
                    
                while mag>max_mag:
                    width = int(np.round(width/2))
                    height = int(np.round(height/2))
                    mag = mag/2
                    pixel_size = pixel_size*2
                    
                image= scene.read_block(scene.rect,(width,height))
                image_data_bw = image.min(axis=2)
                
                information['ImageName'] = file
                information['Magnification'] = mag
                information['ImagePixelHeight'] = height
                information['ImagePixelWidth'] = width
                information['PixelSizeMeters'] = pixel_size
                filetowrite.write(str(information))
                filetowrite.write(' \n ')
                filetowrite.write('#####')         
                filetowrite.write(' \n ')
    
                object_h = self.ObjectSplitter(image_data_bw, axis = 0)
                width,height=np.shape(image_data_bw)
                for j,indices_v in enumerate(object_h):
                    real_objects = self.ObjectSplitter(image_data_bw[0:height,indices_v[0]:indices_v[1]], axis = 1)
                    
                    for k,indices_h in enumerate(real_objects):
                        full_path = os.path.join(img_fold, file+"_"+str(j)+"_mag"+str(int(mag))+".png")
                        if not os.path.exists(full_path):
                            self.BackgroundReducer(image[indices_h[0]:indices_h[1],indices_v[0]:indices_v[1]],  full_path)  
        bar.finish()
                            

    #return the indices splitting pairs of an image array depending on the percentage that of pixel 'completion' along a given axis.
    #This is to say, this function generates indices that an imaged should be cropped between,
    #either vertically or horziontally, based upon the percentage of white background in the image
    #the default percentage is 2%
    def ObjectSplitter(self, image_arr,percentage_threshold=2, axis =0):
        
        val = filters.threshold_otsu(image_arr)
        data = np.sum(image_arr < val,axis=axis)
        n = data.shape[0]
        data = np.where(data<(percentage_threshold/100)*n,0,1)
        
        loc_run_start = np.empty(n, dtype=bool)
        loc_run_start[0] = True
        np.not_equal(data[:-1], data[1:], out=loc_run_start[1:])
        run_starts = np.nonzero(loc_run_start)[0].tolist()

        # find run values
        run_values = data[loc_run_start].tolist()

        # find run lengths
        run_lengths = np.diff(np.append(run_starts, n)).tolist()
        
        counter = 0
        
        for i in range(len(run_starts)):
            idx = i-counter
            if run_lengths[idx]<0.02*n:
                if idx==0:continue
                if run_lengths[idx-1]>0.05*n and run_values[idx-1]==1:
                    run_lengths[idx-1] += run_lengths[idx]
                    run_lengths.pop(idx)
                    run_starts.pop(idx)
                    run_values.pop(idx)
                    counter+=1
                    continue
                    
        
                if idx>=len(run_starts)-1:continue
                if run_lengths[idx+1]>0.05*n and run_values[idx+1]==1:
                    run_lengths[idx+1] += run_lengths[idx]
                    run_lengths.pop(idx)
                    run_starts.pop(idx)
                    run_values.pop(idx)
                    counter+=1
                    continue
                
                run_lengths[idx-1] += run_lengths[idx]
                run_lengths.pop(idx)
                run_starts.pop(idx)
                run_values.pop(idx)
                counter+=1
               
        object_pairs = []
        for i in range(len(run_values)):
            if not run_values[i]==1:continue
            object_pairs.append((run_starts[i], run_starts[i]+run_lengths[i]))
        
        return object_pairs
        
        
    #this function uses the indices splitting pairs to split input images.
    def BackgroundReducer(self, image, png_save, true_boundary = 0.01):
        
        def consecutive(data, stepsize=1):
            arr_consec= np.split(data, np.where(np.diff(data) != stepsize)[0]+1)
            return max(arr_consec, key = len) 
        
        image_data_bw = image.min(axis=2)
        

        gc.collect()
        non_empty = np.where(image_data_bw<220,True, False)
        non_empty_columns = np.where(np.sum(non_empty,axis=0)>true_boundary*np.shape(non_empty)[1])
        non_empty_rows = np.where(np.sum(non_empty,axis=1)>true_boundary*np.shape(non_empty)[0])
        
        non_empty_cols_consec = consecutive(non_empty_columns)
        non_empty_rows_consec = consecutive(non_empty_rows) 
        
        try:
            cropBox = (np.min(non_empty_rows_consec), np.max(non_empty_rows_consec), np.min(non_empty_cols_consec), np.max(non_empty_cols_consec))
        except:
            print("Improper Object found. Moving on...")
            gc.collect()
            return
        Image.fromarray(image[cropBox[0]:cropBox[1]+1, cropBox[2]:cropBox[3]+1 , :]).save(png_save)

        gc.collect()
        
                
if __name__ == '__main__':
    a= DataPreparation("C:\\Users\\Billy\\Downloads\\Data")
    time.sleep(2)
    a.AutocropAll()
        