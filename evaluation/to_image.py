import pandas as pd
import os 
import matplotlib.pyplot as plt


#transform a pandas dataframe into an array of images of size w*h*c 
def to_image(df,w,h,c):
    '''
    w : image width
    h : image height
    c : number of channels    
    '''
    images = df.iloc[:,0:].values.reshape(len(df),w,h,c)
    return images
