import tensorflow as tf
import os
import glob
import numpy as np
import shutil
from patchify import patchify, unpatchify
import random
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm  #Progress bar for loops 
#from skimage.io import imread, imshow
#from skimage.color import rgb2gray
#from skimage.transform import resize
import random
import numpy as np
import ezdxf , glob

#Plotting 
def random_plot(images,masks,N,ftype="files"):
    for _ in range(N):
        image_number = random.randint(0, len(images))
        if ftype=="files":
            img=cv2.imread(images[image_number])
            label=cv2.imread(masks[image_number],0)
        else:
            img=images[image_number]
            label=masks[image_number ]
            
        print(img.shape)
        plt.figure(figsize=(12, 6))
        plt.subplot(121)
        plt.imshow(img)
        plt.subplot(122)
        plt.imshow(label)
        plt.show()

#Plotting Model output results 
def compare_TV(history,savepath,modelname,BATCH_SIZE,EPOCHS):
      #  X_test,y_test=val_img_gen.__next__()
        import matplotlib.pyplot as plt
        if not os.path.exists(savepath):
              os.makedirs(savepath)
        keys=list(history.history.keys())
        VAL=False
        if "val_loss" in keys:
            VAL=True
            keys=list([k for k in keys if "val" not in k])
  
        for key in keys:
            score=history.history[key]
            epochs=range(len(score))
            
            plt.plot(epochs, score, 'bo' ,label = f'training {key}')
            if VAL:
                plt.plot(epochs,history.history[f"val_{key}"], 'b' , label= f'validation {key}')
            
            plt.title(modelname+ "\n"+key)
            plt.legend()
            plt.savefig(f'{savepath}/{key}_{modelname}_{EPOCHS:03d}_{BATCH_SIZE:03d}.png')
            plt.figure()



#Patching and file processing 
#X : Vectors (images) Y : labels (location boolien)
def Readfiles(files,GRAY=False,Patch=False,patch_size=128):
    out_images=[]
    for n,f in tqdm(enumerate(files),total=len(files)):
    #img = imread(f)[:,:,:IMG_CHANNELS] tqdm show a smart progress bar 
        if GRAY:
            img=cv2.imread(f,0)
            c=1
        else:
            img=cv2.imread(f,1)
            c=3
        if Patch:
            SIZE_X = (img.shape[1]//patch_size)*patch_size #Nearest size divisible by our patch size
            SIZE_Y = (img.shape[0]//patch_size)*patch_size #Nearest size divisible by our patch size
            #print(SIZE_X)
            #print(SIZE_Y)
            if GRAY:
                img=img[0:SIZE_Y,0:SIZE_X]
                patches_img = patchify(img, (patch_size, patch_size), step=patch_size)  #Step=256 for 256 patches means no overlap
                dim_p=patches_img.shape
                #print(dim_p)
                patches_img=patches_img.reshape(np.prod(dim_p[0:2]),dim_p[2],dim_p[3])
            else:
                img=img[0:SIZE_Y,0:SIZE_X,:]
                patches_img = patchify(img, (patch_size, patch_size, 3), step=patch_size)  #Step=256 for 256 patches means no overlap
                dim_p=patches_img.shape
                patches_img=patches_img.reshape(np.prod(dim_p[0:3]),dim_p[3],dim_p[4],dim_p[5])
                
                
            for img in patches_img:
                out_images.append(img)
            #np.sum(patches.shape[0:3])
            #return patches_img
        else:    
            out_images.append(img)
    #print( dim_p)
    return np.array(out_images)

def UsefullImageFilter(train_imgs,train_labels,USEFULL_THRESH=100): 
#Checks amounts of pixels in that countain labels. Use to cut blank images
    
    X_train=[]  
    y_train=[]
    for i in range(len(train_labels)):
        val,counts=np.unique(train_labels[i],return_counts=True)
        if len(val)>1:
            if counts[1]>USEFULL_THRESH:
                y_train.append(train_labels[i])
                X_train.append(train_imgs[i])
    return np.array(X_train),np.array(y_train)

def PrepTraningData(train_imgs_loc,train_label_loc,
    show=False,Patch=True,patch_size=256,GRAY=False,
    USEFULL=True,USEFULL_THRESH=150
):
    #Get images and labels 
    train_files=glob.glob(train_imgs_loc+"/***.jpg")
    train_labels=glob.glob(train_label_loc+"/***.png")
    #Sort images 
    train_labels=sorted(train_labels) #to be sure that images and label have the same order
    train_files=sorted(train_files)
   
    if show:
        random_plot(train_files,train_labels,5)

    
    train_I=Readfiles(train_files,Patch=Patch,patch_size=patch_size,GRAY=GRAY) #if GRAY only chage RBG imges
    train_L=Readfiles(train_labels,Patch=True,patch_size=patch_size,GRAY=True) #Lables always gray
    #Added the label channel to labels 
    dim_L=train_L.shape
    train_L=train_L.reshape(dim_L[0],dim_L[1],dim_L[2],1)

    if show:
        random_plot(train_I,train_L[:,:,:,0],6,ftype="array")

    #Usefull data 
    if USEFULL:
        train_I,train_L=UsefullImageFilter(train_I,train_L,USEFULL_THRESH=USEFULL_THRESH)
    else:
        train_I,train_L=np.array(train_I),np.array(train_L)
    if show:
        print("Finnal data")
        print("Images",train_I.shape)
        print("Labels",train_L.shape)
    return train_I,train_L
