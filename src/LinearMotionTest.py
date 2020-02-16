#This is the linear motion test
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as ss
import skimage
from numba import jit
from skimage.viewer import ImageViewer
from skimage.restoration import richardson_lucy
import os
import copy




@jit #Numba jit compiler
def circular_gaussian(mean,sigma,kdim):
    x = np.linspace(-2*sigma,2*sigma,kdim)
    y = np.linspace(-2*sigma,2*sigma,kdim)
    xx,yy = np.meshgrid(x,y)
    kernel = (1/(2*np.pi*sigma**2))*np.exp(-0.5*np.power((xx-mean)/sigma,2) - 0.5*np.power((yy-mean)/sigma,2))
    kernel = kernel/np.sum(kernel) #Normalize Kernel
    return xx,yy,kernel

@jit
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
            plt.savefig("rec_color{}".format(i))
            return


class Image:
    def __init__(self,name,direc = 'images'):
        self.direc = direc
        params = name.split('_')
        self.trial_num = params[0]
        self.shutter_speed = int(params[1])
        self.direction = params[2]
        speed = params[3].lower()
        self.true_speed = int(speed.replace('.jpg',''))
        self.data = plt.imread(direc + "/%s"%name)
        return

    def recover(self, widths = (5,20)):
        self.data = skimage.transform.resize(self.data,(800,1200)) #Downsamples image by a factor of 1
        tesla_orig = copy.deepcopy(self.data)
        tesla = copy.deepcopy(tesla_orig)
        tesla = tesla[320:490,0:1200]
        R = tesla[:,:,0]; G = tesla[:,:,1]; B = tesla[:,:,2];
        new_directory_name = 'Trial{}_Shutter{}_Speed{}'.format(self.trial_num,self.shutter_speed,self.true_speed)
        os.mkdir(new_directory_name)
        for i in range(widths[0],widths[1] + 1): #Recovery Iteration
            print('Kernel Size: {}'.format(i))
            kernel_motion_blur = LinearKernel(size = i,dir= self.direction)
            recoveredR = Richardson_Iteration(R,kernel_motion_blur,40)
            recoveredG = Richardson_Iteration(G,kernel_motion_blur,40)
            recoveredB = Richardson_Iteration(B,kernel_motion_blur,40)
            recoveredc = np.empty((recoveredR.shape[0],recoveredR.shape[1],3))
            recoveredc[:,:,0] = recoveredR; recoveredc[:,:,1] = recoveredG; recoveredc[:,:,2] = recoveredB
            recoveredc = channelnorm(recoveredc,0, vmin = R.min(), vmax = R.max())
            recoveredc = channelnorm(recoveredc,1, vmin = G.min(), vmax = G.max())
            recoveredc = channelnorm(recoveredc,2, vmin = B.min(), vmax = B.max())
            # tesla_temp = tesla_orig
            # tesla_temp[150:250,0:600] = recoveredc
            plt.imshow(recoveredc)
            plt.title("Recovered for {} x {} kernel".format(i,i))
            fname = 'rec_color_{}'.format(i)
            plt.savefig(new_directory_name + '/' + fname)
        return
if __name__ == '__main__':



    for image in os.listdir('images'):
        im_obj = Image(image,'images')
        im_obj.recover(widths=(5,60))

