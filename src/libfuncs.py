#Bayesian Image Restoration
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.image as img
from skimage import color, data, restoration
from mpl_toolkits.mplot3d import Axes3D
import scipy.signal as ss
import time
from numba import jit







    #My function
def Richardson_Iteration(H,kernel,num_its,convolve_method = 0):
    #If convolve_method = 0, we will use the FFT Method
    #Otherwise we will use the normal 2d convolve
    #First compute parameters
    S = kernel #makes life easier
    (N,M) = H.shape #Dimensions of the degraded image
    K,L = S.shape #Size of the Kernel
    I = N - K + 1 # Dimensions of the original image
    J = M - K + 1
    II = np.sum(H) # Total energy of degraded image
    W = .5*np.ones((N,M,num_its)) #This will force the initial condition as the first iteration
    for r in range(num_its-1):
        if convolve_method == 0: 
            temp0 = H/ss.fftconvolve(W[:,:,r],kernel,'same')
            temp1 = ss.fftconvolve(temp0,S[::-1,::-1],'same') #Reverse the Kernel
            W[:,:,r+1] = W[:,:,r]*temp1
        
        else:
            temp0 = H/ss.convolve2d(W[:,:,r],kernel,'same')
            temp1 = ss.convolve2d(temp0,S[::-1,::-1],'same') #Reverse the Kernel
            W[:,:,r+1] = W[:,:,r]*temp1
    return W


#Faster Function, Does not store each iteration
def Richardson_Iteration2(H,kernel,num_its,convolve_method = 0):
    #If convolve_method = 0, we will use the FFT Method
    #Otherwise we will use the normal 2d convolve
    #First compute parameters
    S = kernel #makes life easier
    (N,M) = H.shape #Dimensions of the degraded image
    K,L = S.shape #Size of the Kernel
    I = N - K + 1 # Dimensions of the original image
    J = M - K + 1
    II = np.sum(H) # Total energy of degraded image
    W = .5*np.ones((N,M)) #This will force the initial condition as the first iteration
    if convolve_method == 0:
        
        for r in range(num_its):
            temp0 = H/ss.fftconvolve(W,kernel,'same')
            temp1 = ss.fftconvolve(temp0,S[::-1,::-1],'same') #Reverse the Kernel
            W = W *temp1
        
    else:
        for r in range(num_its):
            temp0 = H/ss.convolve2d(W,kernel,'same')
            temp1 = ss.convolve2d(temp0,S[::-1,::-1],'same') #Reverse the Kernel
            W = W*temp1
    return W

    #Image Functions
@jit
def im2bw(image): #Black and White Imag
    return 0.2126*image[:,:,0] + 0.7152*image[:,:,1] + 0.0722*image[:,:,2]
    
@jit
def blur_image(image,kdim,sigma=1): #Blurring Function
    xx,yy,kernel = circular_gaussian(0,sigma,kdim)
    return ss.fftconvolve(image,kernel),kernel

def blur_image_color(image,kdim,sigma=1): #Blurring Function
    xx,yy,kernel = circular_gaussian(0,sigma,kdim)
    n,m,l = image.shape
    blurred_image = np.empty((n + kdim - 1, m + kdim - 1, 3))
    blurred_image[:,:,0] = ss.fftconvolve(image[:,:,0],kernel)
    blurred_image[:,:,1] = ss.fftconvolve(image[:,:,1],kernel)
    blurred_image[:,:,2] = ss.fftconvolve(image[:,:,2],kernel)
    return blurred_image,kernel


def deblur_color(blurry,kernel,num_its): 
    W = np.empty(blurry.shape)
    W[:,:,0] = Richardson_Iteration(blurry[:,:,0],kernel,num_its)[:,:,-1]
    W[:,:,1] = Richardson_Iteration(blurry[:,:,1],kernel,num_its)[:,:,-1]
    W[:,:,2] = Richardson_Iteration(blurry[:,:,2],kernel,num_its)[:,:,-1]
    return W

def deblur_color2(blurry,kernel,num_its): 
    W = np.empty(blurry.shape)
    W[:,:,0] = restoration.richardson_lucy(blurry[:,:,0],kernel,num_its)
    W[:,:,1] = restoration.richardson_lucy(blurry[:,:,1],kernel,num_its)
    W[:,:,2] = restoration.richardson_lucy(blurry[:,:,2],kernel,num_its)
    return W