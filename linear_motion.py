#This is the linear motion test
import numpy as np
import matplotlib.pyplot as plt 
import scipy.signal as ss
import skimage 
from numba import jit 
from skimage.viewer import ImageViewer
from skimage.restoration import richardson_lucy
import os




@jit #Numba jit compiler
def circular_gaussian(mean,sigma,kdim):
    x = np.linspace(-2*sigma,2*sigma,kdim)
    y = np.linspace(-2*sigma,2*sigma,kdim)
    xx,yy = np.meshgrid(x,y)
    kernel = (1/(2*np.pi*sigma**2))*np.exp(-0.5*np.power((xx-mean)/sigma,2) - 0.5*np.power((yy-mean)/sigma,2))
    kernel = kernel/np.sum(kernel) #Normalize Kernel
    return xx,yy,kernel

def channelnorm(im, channel, vmin, vmax):
    c = (im[:,:,channel]-vmin) / (vmax-vmin)
    c[c<0.] = 0
    c[c>1.] = 1
    im[:,:,channel] = c
    return im


#Faster Function, Does not store each iteration
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
    W = (1/N)*np.ones((N,M)) #This will force the initial condition as the first iteration
    if convolve_method == 0:
        for r in range(num_its):
            g = ss.fftconvolve(W,kernel,'same')
#           g[g == 0 ] = 10**(-15)
            temp0 = H/g
            temp1 = ss.fftconvolve(temp0,S[::-1,::-1],'same') #Reverse the Kernel
            W = W * temp1
            print("Iteration: %d"%r)
        
    else:
        # import pdb; pdb.set_trace()
        for r in range(num_its):
            temp0 = H/ss.convolve2d(W,kernel,'same')
            temp1 = ss.convolve2d(temp0,S[::-1,::-1],'same') #Reverse the Kernel
            W = W*temp1
    return W


    
def LinearKernel(size,dir='right'):
    kernel_motion_blur = np.zeros((2*size+1, 2*size+1))
    kernel_motion_blur[(size), (size)::] = np.ones(size+1)
    kernel_motion_blur[size,size] = 1
    kernel_motion_blur = kernel_motion_blur /(size + 1)
    if dir == 'right':
        return kernel_motion_blur
    else: 
        return np.flip(kernel_motion_blur)


#Recovery Functions
    def restore_image(fname, scale_factor, num_its = 15, direction = 'left',range=(5,20)):


        image = plt.imread(fname) #reads image
        w = image.shape[0]; l = image.shape[1]
        wscale = int(image.shape[0]/scale_factor); lscale = int(image.shape[1]/scale_factor); 
        image = skimage.transform.resize(image,(400,600)) #Downsamples image by a factor of 10
        R = image[:,:,0]; G = image[:,:,1]; B = image[:,:,2]; #Splits RGB Channels
        R = R/R.max(); G = G/G.max(); B = B/B.max() #normalizes each channel by its maximum value
        for i in range(5,20): #Recovery Iteration
            print('Kernel Size: {}'.format(i))
            kernel_motion_blur = LinearKernel(size = i,dir='left')
            recoveredR = Richardson_Iteration(R,kernel_motion_blur,20)          
            recoveredG = Richardson_Iteration(G,kernel_motion_blur,20)
            recoveredB = Richardson_Iteration(B,kernel_motion_blur,20)
            recoveredc = np.empty((recoveredR.shape[0],recoveredR.shape[1],3))
            recoveredc[:,:,0] = recoveredR; recoveredc[:,:,1] = recoveredG; recoveredc[:,:,2] = recoveredB
            recoveredc = channelnorm(recoveredc,0, vmin = R.min(), vmax = R.max())
            recoveredc = channelnorm(recoveredc,1, vmin = G.min(), vmax = G.max())
            recoveredc = channelnorm(recoveredc,2, vmin = B.min(), vmax = B.max())
            np.save("recoveredc{}.format(i)",recoveredc)
            plt.figure()
            plt.imshow(recoveredc)
            plt.title("Recovered for {} x {} kernel".format(i,i))
            plt.savefig("rec_color{}".format(i))
            return
if __name__ == '__main__':
    
    ################################################################################
##    linimage_color = plt.imread('og.jpg').astype(np.float64)
##    linimage_color = skimage.transform.resize(linimage_color,(400,600))
##    #Color
##    kernel_motion_blur = LinearKernel(size = i,dir='right')
##    Rblur  = ss.fftconvolve(linimage_color[:,:,0], kernel_motion_blur)
##    Gblur  = ss.fftconvolve(linimage_color[:,:,1], kernel_motion_blur)
##    Bblur  = ss.fftconvolve(linimage_color[:,:,2], kernel_motion_blur) 
##    imblur_color = np.empty((Rblur.shape[0],Rblur.shape[1],3))
##    imblur_color[:,:,0] = Rblur/Rblur.max();
##    imblur_color[:,:,1] = Gblur/Gblur.max();
##    imblur_color[:,:,2] = Bblur/Bblur.max();
    ################################################################################


    ################################################################################
#    linimage_color = plt.imread('tesla.jpg').astype(np.float64)
#    teslabw = (0.2989*linimage_color[:,:,0] + 0.5870*linimage_color[:,:,1] + 0.1140*linimage_color[:,:,2])
#    teslabw= skimage.transform.resize(teslabw,(400,600))
#    theta = 0
#    size = 10
#    for i in range(5,20):
#        print('Kernel Size: {}'.format(i))
#        kernel_motion_blur = LinearKernel(size = i,dir='left')
#        recovered = Richardson_Iteration(teslabw,kernel_motion_blur,20)
#        np.save("recovered{}.format(i)",recovered)
#        plt.figure()
#        plt.imshow(recovered,cmap = 'gray',vmin = teslabw.min(), vmax = teslabw.max())
#        plt.title("Recovered for {} x {} kernel".format(i,i))
#        plt.savefig("rec{}".format(i))
    ################################################################################
    


    ################################################################################
    tesla = plt.imread('full.jpg') #reads image
    tesla= skimage.transform.resize(tesla,(800,1200)) #Downsamples image by a factor of 10
    R = tesla[:,:,0]; G = tesla[:,:,1]; B = tesla[:,:,2]; #Splits RGB Channels
    R = R/R.max(); G = G/G.max(); B = B/B.max() #normalizes each channel by its maximum value

    for i in range(10,55): #Recovery Iteration
        print('Kernel Size: {}'.format(i))
        kernel_motion_blur = LinearKernel(size = i,dir='left')
        recoveredR = Richardson_Iteration(R,kernel_motion_blur,40)
        recoveredG = Richardson_Iteration(G,kernel_motion_blur,40)
        recoveredB = Richardson_Iteration(B,kernel_motion_blur,40)
        recoveredc = np.empty((recoveredR.shape[0],recoveredR.shape[1],3))
        recoveredc[:,:,0] = recoveredR; recoveredc[:,:,1] = recoveredG; recoveredc[:,:,2] = recoveredB
        recoveredc = channelnorm(recoveredc,0, vmin = R.min(), vmax = R.max())
        recoveredc = channelnorm(recoveredc,1, vmin = G.min(), vmax = G.max())
        recoveredc = channelnorm(recoveredc,2, vmin = B.min(), vmax = B.max())
        np.save("recoveredc{}.format(i)",recoveredc)
        plt.figure()
        plt.imshow(recoveredc)
        plt.title("Recovered for {} x {} kernel".format(i,i))
        plt.savefig("full/rec_color{:}".format(i))
    ################################################################################

