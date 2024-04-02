import PySimpleGUI as sg
import tkinter as tk
from tkinter import filedialog
from tkinter.filedialog import asksaveasfile

import numpy as np 
import pandas as pd
import os
import os.path
import glob
import pathlib
import cv2
from skimage.segmentation import watershed
import copy
import os
import shutil
import keras
import tensorflow

# reference folder global
ref_folder = ""

def store_images(image, image_id):
    """ Stores a single image to an HDF5 file.
        Parameters:
        ---------------
        image       image array, (32, 32, 3) to be stored
        image_id    integer unique ID for image
     
    """
    global ref_folder

    
    
    path = os.getcwd()
    ref_folder = os.path.join(path, "reference_folder")
    if not os.path.exists(ref_folder):
        os.makedirs(ref_folder)
    else:
        ref_folder=ref_folder
    #image.save(os.path.join(ref_folder, image_id))
    cv2.imwrite(os.path.join(ref_folder, image_id), image)
    return(ref_folder)

def cziFileNameCut(fName):
    key = ".czi - "
    try:
        index = fName.index(key)
        return fName[index + len(key):].strip()
    except ValueError:
        # no .czi, just return the name as is
        return fName

def analyze_image(im_path):
    
    
    '''
    Take an image_path (pathlib.Path object), preprocess and label it, extract the RLE strings 
    and dump it into a Pandas DataFrame.
    '''
    # Read in data and convert to grayscale
    im_id = cziFileNameCut(os.path.basename(im_path))
    im = cv2.imread(str(im_path))
    
    
    store_images(image=im,image_id=im_id)
    #COnverting to grayscale
    im_gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    #im1 = cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
    im1=im
    
    im_blur=cv2.GaussianBlur(im_gray,(5,5),0)
    import matplotlib.pyplot as plt

    
    
    ret,th = cv2.threshold(im_blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    #plt.figure(figsize=(10,10))
    #plt.subplot(121)
    #plt.imshow(th,cmap='gray')
    #plt.axis("off")
    #plt.show()
    
    # noise removal
 
    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(th,cv2.MORPH_OPEN,kernel, iterations = 3)
    #plt.figure(figsize=(10,10))
    #plt.subplot(121)
    #plt.imshow(opening,cmap='gray')
    #plt.axis("off")
    #plt.show()
    
    # sure background area
    sure_bg = cv2.dilate(opening,kernel,iterations=3)
    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
    ret, sure_fg = cv2.threshold(dist_transform,0.005*dist_transform.max(),255,0)
    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg,sure_fg)
    # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)
    # Add one to all labels so that sure background is not 0, but 1
    markers = markers+1
    # Now, mark the region of unknown with zero
    markers[unknown==255] = 0
    #plt.figure(figsize=(10,10))
    #plt.subplot(121)
    #plt.imshow(markers,cmap='jet')
    #plt.axis("off")
    #plt.show()
    markers = cv2.watershed(im1,markers)
    im1[markers == -1] = [255,0,0]
    #fig=plt.figure(figsize=(20, 20), dpi= 80, facecolor='w', edgecolor='k')
    #plt.axis("off")
    #plt.subplot(131)
    #plt.imshow(im1)
    #plt.axis("off")
    #plt.subplot(132)
    #plt.imshow(markers,cmap='gray')
    #plt.axis("off")
    #plt.show()
    #plt.subplot(133)
    #plt.imshow(im_gray,cmap='gray')
    #plt.axis("off")
    #plt.show()
    mask = np.where(markers > sure_fg, 1, 0)

    # Make sure the larger portion of the mask is considered background
    if np.sum(mask==0) < np.sum(mask==1):
        mask = np.where(mask, 0, 1)
        
    from scipy import ndimage
    labels, nlabels = ndimage.label(mask)
    #print('There are {} separate components / objects detected.'.format(nlabels))
    
    for label_ind, label_coords in enumerate(ndimage.find_objects(labels)):
         cell = markers[label_coords]
    
         # Check if the label size is too small
         if np.product(cell.shape) < 10: 
             #print('Label {} is too small! Setting to 0.'.format(label_ind))
             mask = np.where(labels==label_ind+1, 0, mask)

    # Regenerate the labels
    labels, nlabels = ndimage.label(mask)

    label_arrays = []
    for label_num in range(1, nlabels+1):
        label_mask = np.where(labels == label_num, 1, 0)
        label_arrays.append(label_mask)
    
    #print('There are now {} separate components / objects detected.'.format(nlabels))
    
    
    im_df = pd.DataFrame()
    for label_num in range(1, nlabels+1):
        label_mask = np.where(labels == label_num, 1, 0)
        if label_mask.flatten().sum() > 10:

            s = pd.Series({'ImageId': im_id,'Nuclei Count':nlabels})
            
                                       
            im_df = pd.concat([im_df,s],axis=1)
    
    return im_df


def analyze_list_of_images(im_path_list):
    
    #Takes a list of image paths (pathlib.Path objects), analyzes each,
    #and returns a submission-ready DataFrame, as well a refernce folder.
    
    all_df = pd.DataFrame()
    for im_path in im_path_list:
        im_df = analyze_image(im_path)
        all_df = pd.concat([im_df,all_df],axis=1)
    
    return all_df


# analyze_list_of_images(pathlib.Path('c:/Users/ccape/Downloads/nucelui data').glob('*.tif'))

#front end
sg.theme('Reddit')
layout =  [ [sg.Text("Count Nuclei in a folder of images: "), sg.In(key='-USER FOLDER-'), sg.FolderBrowse(target='-USER FOLDER-'),[sg.Submit()],[sg.Cancel()]]]
        
newlayout = copy.deepcopy(layout)
window = sg.Window('Select and submit folder of images', newlayout, size=(270*4,4*100))
event, values = window.read()
image_path = None
while True:
    event, values = window.read()
    print(event, values)
    
    if event == 'Cancel':
        break
    elif event == 'Submit':
        #results=analyze_list_of_images("im_path_list")
        #results
        image_path = values["-USER FOLDER-"]
        break
window.close()
if image_path:
    image_list = glob.glob(os.path.join(image_path, "*.tif"))
    results = analyze_list_of_images([pathlib.Path(s) for s in image_list])
#widget = qgrid.show_grid(results)
#widget

        
if results is not None:        
        my_w = tk.Tk()
        my_w.geometry("400x300")  # Size of the window 
        my_w.title('Save results as a CVS')
        my_font1=('times', 18, 'bold')
        l1 = tk.Label(my_w,text='Save File',width=30,font=my_font1)
        l1.grid(row=1,column=1)
        
        b1 = tk.Button(my_w, text='Save', 
        width=20,command = lambda:save_file())
        b1.grid(row=2,column=1)
        
        l2 = tk.Label(my_w,text='Save Folder',width=30,font=my_font1)
        l2.grid(row=3,column=1)
        b2 = tk.Button(my_w, text='Save', 
        width=20,command = lambda:save_folder())
        b2.grid(row=4,column=1)
        def save_file():
            file = filedialog.asksaveasfilename(
                
            filetypes=[("csv file", ".cvs")],
            defaultextension=".csv",
            title='Save Output')
            results_file=results.to_csv(str(file))
            if file: 
                            fob=open(str(results_file),'w')
                            fob.write("Save results")
                            fob.close()
            else: # user cancel the file browser window
                        print("No file chosen")
        def save_folder():
            print("Save folder pressed: ", ref_folder)
            destFolder = filedialog.askdirectory()
            print("Destination folder: ", destFolder)
            if (os.path.exists(ref_folder) and os.path.exists(destFolder)):
                shutil.copytree(ref_folder, destFolder, dirs_exist_ok=True)
            
        my_w.mainloop()  # Keep the window open
""" if os.path.exists(ref_folder):
    my_w = tk.Tk()
    my_w.geometry("400x300")  # Size of the window 
    my_w.title('Save reference folder')
    my_font1=('times', 18, 'bold')
    l1 = tk.Label(my_w,text='Save Folder',width=30,font=my_font1)  
    l1.grid(row=1,column=1)
    b1 = tk.Button(my_w, text='Save', 
      width=20,command = lambda:save_file())
    b1.grid(row=2,column=1) 
    def save_folder():
        dir_name = tk.tkFileDialog.askdirectory(title='Save folder')
        if dir_name:
            pass 
            #fob=open(str(results_file),'w')
            #fob.write("Save results")
            #fob.close()
        else: # user cancel the file browser window
            print("No file chosen") 
        my_w.mainloop()  # Keep the window open
    """