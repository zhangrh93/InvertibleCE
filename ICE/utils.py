'''
@Author: Zhang Ruihan
@Date: 2019-10-28 01:01:52
@LastEditors  : Zhang Ruihan
@LastEditTime : 2020-01-15 05:50:01
@Description: file content
'''
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
from skimage.transform import resize


#npdir = '/dataset/ILSVRC2012/nparray_train'
#imdir = '/dataset/ILSVRC2012/ILSVRC2012_img_train'

mean = [103.939, 116.779, 123.68]
SIZE = [224,224]
EPSILON = 1e-8

class img_utils():

    def __init__(self,
                img_size = (224,224),
                nchannels = 3,
                img_format = "channels_last",
                mode = None,
                std = None,
                mean = None):
        self.img_format = img_format
        self.nchannels = nchannels
        self.fsize = list(img_size)
        self.img_size = self.fsize + [self.nchannels]
        #if img_format == 'channels_first':
        #    self.img_size = [self.nchannels] + self.fsize
        #else:
        #    self.img_size = self.fsize + [self.nchannels]
        
        self.std = std
        self.mean = mean
        self.mode = mode


    def deprocessing(self,x):
        mode = self.mode
        x = np.array(x)
        X = x.copy()

        if self.img_format == "channels_first":
            if X.ndim == 3:
                X = np.transpose(X,(1,2,0))
            else:
                X = np.transpose(X,(0,2,3,1))

        if mode is None:
            return X

        if mode == "tf":
            X +=1
            X *=127.5
            return X

        if mode == "torch":
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]

        if mode == 'caffe':
            mean = [103.939, 116.779, 123.68]
            std = None            

        if mode == "norm":
            mean = self.mean
            std = self.std


        if std is not None:
            X[..., 0] *= std[0]
            X[..., 1] *= std[1]
            X[..., 2] *= std[2]        
        X[..., 0] += mean[0]
        X[..., 1] += mean[1]
        X[..., 2] += mean[2]   
            
        if mode == 'caffe':
            # 'RGB'->'BGR'
            X = X[..., ::-1]  
        
        if mode == "torch":
            X *= 255
        return X
    
    def resize_img(self,array, smooth = False):
        fsize = self.fsize
        size = array.shape
        if smooth:            
            tsize = list(size)
            tsize[1] = fsize[0]
            tsize[2] = fsize[1]
            res = resize(array, tsize, order=1, mode='reflect',anti_aliasing=False)
        else:
            res = []
            for i in range(size[0]):
                temp = array[i]
                temp = np.repeat(temp,fsize[0]//size[1],axis = 0)
                temp = np.repeat(temp,fsize[1]//size[2],axis = 1)
                res.append(temp)
            res = np.array(res)
        return res
    

    def flatten(self,array):
        size = array.shape
        return array.reshape(-1,size[-1])  

    def show_img(self,X,nrows=1,ncols=1,heatmaps = None,useColorBar = True, deprocessing = True, save_path = None):
        X = np.array(X)
        if not heatmaps is None:
            heatmaps = np.array(heatmaps)
        if len(X.shape)<4:
            print ("Dim should be 4")
            return 

        X = np.array(X)
        if deprocessing:
            X = self.deprocessing(X)

        if (not X.min() == 0) or X.max()>1:
            X = X - X.min()
            X = X / X.max()

        if X.shape[0] == 1:
            X = np.squeeze(X)
            X = np.expand_dims(X,axis=0)
        else:
            X = np.squeeze(X)

        if self.nchannels == 1:
            cmap = "Greys"
        else:
            cmap = "viridis"
        
        l = nrows*ncols
        plt.figure(figsize=(5*ncols, 5*nrows))
        for i in range(l):
            plt.subplot(nrows, ncols, i+1)
            plt.axis('off')
            img = X[i]
            img = np.clip(img,0,1)
            plt.imshow(img,cmap = cmap)
            if not heatmaps is None:
                if not heatmaps[i] is None:
                    heapmap = heatmaps[i]
                    plt.imshow(heapmap, cmap='jet', alpha=0.5,interpolation = "bilinear")
                    if useColorBar:
                        plt.colorbar()

        if save_path is not None:
            plt.savefig(save_path)
        plt.show()   
        

    def img_filter(self,x,h,threshold=0.5,background = 0.2,smooth = True, minmax = False):
        x = x.copy()
        h = h.copy()

        if minmax:
            h = h - h.min()

        h = h * (h>0)
        for i in range(h.shape[0]):
            h[i] = h[i] / (h[i].max() + EPSILON)

        h = (h - threshold) * (1/(1-threshold))
        
        #h = h * (h>0)

        h = self.resize_img(h,smooth = smooth)
        h = ((h>0).astype("float")*(1-background) + background)
        h_mask = np.repeat(h,self.nchannels).reshape(list(h.shape)+[-1])
        if self.img_format == 'channels_first':
            h_mask = np.transpose(h_mask,(0,3,1,2))
        x = x * h_mask
        
        h = h - h.min()
        h = h / (h.max() + EPSILON)
        
        return x,h


    def contour_img(self,x,h,dpi = 100):
        dpi = float(dpi)
        size = x.shape
        if x.max()>1:
            x = x/x.max()
        #fig = plt.figure(figsize=(size[1]/dpi,size[0]/dpi),dpi=dpi)
        fig = plt.figure(figsize=(size[1]/dpi,size[0]/dpi),dpi=dpi*SIZE[0]/size[0])
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        xa = np.linspace(0, size[1]-1, size[1])
        ya = np.linspace(0, size[0]-1, size[0])
        X, Y = np.meshgrid(xa, ya)
        if x.shape[-1] == 1:
            x = np.squeeze(x)
            ax.imshow(x,cmap = "Greys")
        else:
            ax.imshow(x)
        ax.contour(X, Y, h,colors = 'r')
        return fig

    def res_ana(self,model,classesLoader,classNos,reducer, layer_name = "conv5_block3_out"):

        w,b = model.model.layers[-1].get_weights()
        V_ = reducer._reducer.components_
        w_ = np.dot(V_,w)
        ana = []
        
        for No in classNos:
            target = No
            tX, ty = classesLoader.load_val([No])
            tX = np.concatenate(tX)
            ty = np.concatenate(ty).astype(int)   
            X = model.get_feature(tX,layer_name=layer_name).mean(axis = (1,2))
            S = reducer.transform(X)
            U = X-np.dot(S,V_)

            C1 = np.dot(X,w[:,target])
            C2 = np.dot(S,w_[:,target]) + np.dot(U,w[:,target])

            C = np.dot(S,w_[:,target])
            res = np.dot(U,w[:,target])

            ana.append(np.array([C1,C2,C,res]))
        return [np.array(ana),reducer]