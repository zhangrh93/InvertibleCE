from utils import *
from ClassesLoader import *
from ModelWrapper import *

from ChannelReducer import ChannelReducer, ClusterReducer
import os
import pickle
import ipywidgets as widgets
from ipywidgets import interact
import itertools

import graphviz
import pydotplus
from IPython.display import Image, display

import scipy
import keras
from keras import backend as K

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

import keras.backend.tensorflow_backend as KTF
import tensorflow as tf
from tensorflow.keras.backend import resize_images

FONT_SIZE = 30
CALC_LIMIT = 1e8
TRAIN_LIMIT = 50
TARGET_ERR = 0.2
MIN_STEP = 10
REDUCER_PATH = "reducer/resnet50"
USE_TRAINED_REDUCER = False

class Explainer():
    def __init__(self,
                 title = "",
                 layer_name = "",
                 classesNos = None,
                 utils = None,
                 nchannels = 3,
                 useMean = True,
                 reducer_type = "NMF",
                 n_components = 10,
                 best_n = False,
                 target_err = TARGET_ERR,
                 min_step = MIN_STEP,
                 featuretopk = 20,
                 featureimgtopk = 5,
                 epsilon = 1e-7):
        self.title = title
        self.layer_name = layer_name
        self.classesNos = classesNos
        if self.classesNos is not None:
            self.C2IDX = {c:i for i,c in enumerate(self.classesNos)}
            self.IDX2C = {i:c for i,c in enumerate(self.classesNos)}

        self.useMean = useMean
        self.reducer_type = reducer_type
        self.nchannels = nchannels
        self.featuretopk = featuretopk
        self.featureimgtopk = featureimgtopk #number of images for a feature
        self.n_components = n_components
        self.target_err = target_err
        self.best_n = best_n
        self.min_step = min_step
        self.epsilon = epsilon
        
        self.utils = utils

        self.reducer = None

        self.feature_base = []
        self.features = {}

        self.font = FONT_SIZE
        
    def load(self):
        title = self.title
        with open("Explainers"+"/"+title+"/"+title+".pickle","rb") as f:
            tdict = pickle.load(f)
            self.__dict__.update(tdict)
            
    def save(self):
        if not os.path.exists("Explainers"):
            os.mkdir("Explainers")
        title = self.title
        if not os.path.exists("Explainers"+"/"+title):
            os.mkdir("Explainers"+"/"+title)
        with open("Explainers"+"/"+title+"/"+title+".pickle","wb") as f:
            pickle.dump(self.__dict__,f)

    def train_model(self,model,classesLoader):
        if self.best_n:
            print ("search for best n.")
            self.n_components = self.min_step
            train_count = 0
            while train_count < TRAIN_LIMIT:
                print ("try n_component with {}".format(self.n_components))
                self.reducer = None
                self._train_reducer(model,classesLoader)
                if self.reducer_err.mean() < self.target_err:
                    self._estimate_weight(model,classesLoader)
                    return 
                self.n_components += self.min_step
                train_count += 1
        else:
            self._train_reducer(model,classesLoader)
            self._estimate_weight(model,classesLoader)

    def _train_reducer(self,model,classesLoader):
        X_feature = []

        print ("training reducer:")
        print ("loading data")

        if self.reducer is None:
            if self.reducer_type == "Cluster":
                self.reducer = ClusterReducer(n_clusters = self.n_components)
            elif self.reducer_type == "NMF":
                if len(self.classesNos) == 1 and USE_TRAINED_REDUCER:
                    target_path = REDUCER_PATH + "/{}/{}.pickle".format(self.layer_name,self.classesNos[0])
                    if os.path.exists(target_path):
                        with open(target_path,"rb") as f:
                            reducers = pickle.load(f)
                        if self.n_components in reducers:
                            self.reducer = reducers[self.n_components]['reducer']
                            print ("reducer loaded")
                            
                if self.reducer is None:
                    self.reducer = ChannelReducer(n_components = self.n_components)
            else:
                self.reducer = ChannelReducer(n_components = self.n_components,reduction_alg = self.reducer_type)


        for No,X,y in tqdm(classesLoader.load_train(self.classesNos)):
            featureMaps = model.get_feature(X,self.layer_name)
            X_feature.append(featureMaps)
        #X_feature = np.concatenate(X_feature)
        #X_feature = np.array(X_feature)

        if not self.reducer._is_fit:
            X_feature_f = np.concatenate(X_feature)
            total = np.product(X_feature_f.shape)
            l = X_feature_f.shape[0]
            nX_feature = X_feature_f
            if total > CALC_LIMIT:
                p = CALC_LIMIT / total
                print ("dataset too big, train with {:.2f} instances".format(p))
                idx = np.random.choice(l,int(l*p),replace = False)
                nX_feature = nX_feature[idx]

            print ("loading complete, with size of {}".format(nX_feature.shape))
            start_time = time.time()
            nX = self.reducer.fit(nX_feature)

            print ("reducer trained, spent {} s".format(time.time()-start_time))
        
        self.ncavs = self.reducer._reducer.components_
        
        reX = []
        for i in range(len(self.classesNos)):
            nX = self.reducer.transform(X_feature[i])
            reX.append(self.reducer.inverse_transform(nX))

        err = []
        for i in range(len(self.classesNos)):
            #print (X_feature[0].shape)
            #print (model.target_predict(X_feature[i],layer_name=self.layer_name).shape)
            res_true = model.target_predict(X_feature[i],layer_name=self.layer_name)[:,i] #
            res_recon = model.target_predict(reX[i],layer_name=self.layer_name)[:,i] #
            err.append(abs(res_true-res_recon).mean(axis=0) / res_true.mean(axis=0))


        self.reducer_err = np.array(err)
        if type(self.reducer_err) is not np.ndarray:
            self.reducer_err = np.array([self.reducer_err])

        print ("fidelity: {}".format(self.reducer_err))

        return self.reducer_err

    def _estimate_weight(self,model,classesLoader):
        X_feature = []

        print ("loading data")

        for No,X,y in tqdm(classesLoader.load_train(self.classesNos)):
            featureMaps = model.get_feature(X,self.layer_name)

            X_feature.append(featureMaps)
        X_feature = np.concatenate(X_feature)

        self.test_weight = []
        print ("estimating weight:")
        for i in tqdm(range(self.n_components)):
            ncav = self.ncavs[i,:]

            res1 =  model.target_predict(X_feature - self.epsilon * ncav,layer_name=self.layer_name)
            res2 =  model.target_predict(X_feature + self.epsilon * ncav,layer_name=self.layer_name)

            res_dif = res2 - res1
            dif = res_dif.mean(axis=0) / (2 * self.epsilon)
            if type(dif) is not np.ndarray:
                dif = np.array([dif])
            self.test_weight.append(dif)

        self.test_weight = np.array(self.test_weight)

    def save_features(self,threshold=0.5,background = 0.2,smooth = True):
        feature_path = "Explainers/"+self.title + "/feature_imgs"
        utils = self.utils

        if not os.path.exists(feature_path):
            os.mkdir(feature_path)

        for idx in tqdm(self.features.keys()): 

            x,h = self.features[idx]
            #x = self.gen_masked_imgs(x,h,threshold=threshold,background = background,smooth = smooth)
            minmax = False
            if self.reducer_type == 'PCA':
                minmax = True
            x,h = self.utils.img_filter(x,h,threshold=threshold,background = background,smooth = smooth,minmax = minmax)
            
            nsize = self.utils.img_size.copy()
            nsize[1] = nsize[1]* self.featureimgtopk
            nimg = np.zeros(nsize)
            nh = np.zeros(nsize[:-1])
            for i in range(x.shape[0]):
                timg = utils.deprocessing(x[i])
                if timg.max()>1:
                    timg = timg / 255.0
                    timg = abs(timg)
                timg = np.clip(timg,0,1)
                nimg[:,i*self.utils.img_size[1]:(i+1)*self.utils.img_size[1],:] = timg
                nh[:,i*self.utils.img_size[1]:(i+1)*self.utils.img_size[1]] = h[i]
            fig = self.utils.contour_img(nimg,nh)
            fig.savefig(feature_path + "/"+str(idx)+".jpg",bbox_inches='tight',pad_inches=0)
            plt.close(fig)
            #plt.imsave(feature_path + "/"+str(idx)+".jpg",nimg)

    def feature_filter(self,featureMaps):
        if self.useMean:
            return featureMaps.mean(axis = (1,2))
        else:
            return featureMaps.max(axis=(1,2))
   
    
    def generate_features(self,model,classesLoader, featureIdx = None):
        featuretopk = min(self.featuretopk, self.n_components)

        imgTopk = self.featureimgtopk
        if featureIdx is None:
            featureIdx = []
            tidx = []
            w = self.test_weight
            for i,No in enumerate(self.classesNos):
                tw = w[:,i]
                tidx += tw.argsort()[::-1][:featuretopk].tolist()
            featureIdx += list(set(tidx))                    

        nowIdx = set(self.features.keys())
        featureIdx = list(set(featureIdx) - nowIdx)
        featureIdx.sort()

        if len(featureIdx) == 0:
            print ("All feature gathered")
            return

        print ("generating features:")
        print (featureIdx)

        features = {}
        for No in featureIdx:
            features[No] = [None,None]
        
        print ("loading training data")
        for No,X,y in tqdm(classesLoader.load_train(self.classesNos)):
            
            featureMaps = self.reducer.transform(model.get_feature(X,self.layer_name))
            
            X_feature = self.feature_filter(featureMaps)

            for No in featureIdx:
                samples,heatmap = features[No]
                idx = X_feature[:,No].argsort()[-imgTopk:]
                
                nheatmap = featureMaps[idx,:,:,No]
                nsamples = X[idx,...]
                
                if type(samples) == type(None):
                    samples = nsamples
                    heatmap = nheatmap
                else:
                    samples = np.concatenate([samples,nsamples])
                    heatmap = np.concatenate([heatmap,nheatmap])

                    nidx = self.feature_filter(heatmap).argsort()[-imgTopk:]
                    samples = samples[nidx,...]
                    heatmap = heatmap[nidx,...]
                
                features[No] = [samples,heatmap]
        
        for no,(x,h) in features.items():
            idx = h.mean(axis = (1,2)).argmax()
            for i in range(h.shape[0]):
                if h[i].max() == 0:
                    x[i] = x[idx]
                    h[i] = h[idx]
        
        self.features.update(features)
        self.save()

    def generate_image_LR_file(self,classesLoader):        
        title = self.title
        fpath = os.getcwd() + "\\Explainers\\"+ self.title + "\\feature_imgs\\"
        featopk = min(self.featuretopk,self.n_components)
        imgtopk = self.featureimgtopk
        classes = classesLoader
        Nos = self.classesNos
        fw = self.test_weight

        font = self.font
        
        def LR_graph(wlist,No):
            def node_string(count,fidx,w,No):
                nodestr = ""
                nodestr += "{} [label=< <table border=\"0\">".format(count)

                nodestr+="<tr>"
                nodestr+="<td><img src= \"{}\" /></td>".format(fpath+"{}.jpg".format(fidx)) 
                nodestr+="</tr>"


                #nodestr +="<tr><td><FONT POINT-SIZE=\"{}\"> ClassName: {} </FONT></td></tr>".format(font,classes.No2Name[No])
                nodestr +="<tr><td><FONT POINT-SIZE=\"{}\"> FeatureRank: {} </FONT></td></tr>".format(font,count)

                nodestr +="<tr><td><FONT POINT-SIZE=\"{}\"> Feature: {}, Weight: {:.3f} </FONT></td></tr>".format(font,fidx,w)

                nodestr += "</table>  >];\n" 
                return nodestr

            resstr = "digraph Tree {node [shape=box] ;rankdir = LR;\n"


            count = len(wlist)
            for k,v in wlist:
                resstr+=node_string(count,k,v,No)
                #print (count,k,v)
                count-=1
            
            resstr += "0 [label=< <table border=\"0\">" 
            resstr += "<tr><td><FONT POINT-SIZE=\"{}\"> ClassName: {} </FONT></td></tr>" .format(font,classes.No2Name[No])
            resstr += "<tr><td><FONT POINT-SIZE=\"{}\"> Fidelity error: {:.3f} % </FONT></td></tr>" .format(font,self.reducer_err[self.C2IDX[No]]*100)
            resstr += "<tr><td><FONT POINT-SIZE=\"{}\"> First {} features out of {} </FONT></td></tr>" .format(font,featopk,self.n_components)
            resstr += "</table>  >];\n"
            

            resstr += "}"

            return resstr

        if not os.path.exists("Explainers/"+title+"/GE"):
            os.mkdir("Explainers/"+title+"/GE")
                    
        print ("Generate explanations with fullset condition")

        for i,No in tqdm(enumerate(Nos)):
            wlist = [(j,fw[j][i]) for j in fw[:,i].argsort()[-featopk:]]
            graph = pydotplus.graph_from_dot_data(LR_graph(wlist,No))  
            if not os.path.exists("Explainers/"+title+"/GE/{}.jpg".format(No)):
                graph.write_jpg("Explainers/"+title+"/GE/{}.jpg".format(No))
                


    
                
    def feature_UI(self,heatmapUI = False):
        utils = self.utils

        def view_img(threshold,featureNo):
            samples,heatmap = self.features[featureNo]
            nheatmap = heatmap * (heatmap>=threshold)
            nheatmap = utils.resize_img(nheatmap)
            nsamples = samples * np.repeat(nheatmap,self.nchannels).reshape(list(nheatmap.shape)+[-1])
            utils.show_img(nsamples,1,self.topk)
        return interact(view_img, threshold = (0.0,1.0,0.05), featureNo = list(self.features.keys()))
    

    def local_explanations(self,x,model,classesLoader,target_classes = None,background = 0.2,name = None,with_total = True,display_value = True):
        utils = self.utils
        font = self.font
        featuretopk = min(self.featuretopk,self.n_components)


        if target_classes is None:
            target_classes = self.classesNos
        w = self.test_weight

        pred = model.predict(np.array([x]))[0][target_classes]

        if not os.path.exists("Explainers/"+self.title + "/explanations"):
            os.mkdir("Explainers/"+self.title + "/explanations")

        if not os.path.exists("Explainers/"+self.title + "/explanations/all"):
            os.mkdir("Explainers/"+self.title + "/explanations/all")

        fpath = "Explainers/"+self.title + "/explanations/{}"

        afpath = "Explainers/"+self.title + "/explanations/all/"

        if name is not None:
            if not os.path.exists(fpath.format(name)):
                os.mkdir(fpath.format(name))
            else:
                print ("Folder exists")
                return 
        else:
            count = 0
            while os.path.exists(fpath.format(count)):
                count+=1
            os.mkdir(fpath.format(count))
            name = str(count)

        fpath = fpath.format(name)+"/feature_{}.jpg"

        if self.reducer is not None:
            h = self.reducer.transform(model.get_feature(np.array([x]),self.layer_name))[0]
        else:
            h = model.get_feature(np.array([x]),self.layer_name)[0]

        feature_idx = []
        for cidx in target_classes:
            tw = w[:,self.C2IDX[cidx]]
            tw_idx = tw.argsort()[::-1][:featuretopk]
            feature_idx.append(tw_idx)
        feature_idx = list(set(np.concatenate(feature_idx).tolist()))

        for k in feature_idx:
            
            minmax = False
            if self.reducer_type == "PCA":
                minmax = True

            x1,h1 = utils.img_filter(np.array([x]),np.array([h[:,:,k]]),background=background,minmax = minmax)
            x1 = utils.deprocessing(x1)
            x1 = x1 / x1.max()
            x1 = abs(x1)
            fig = utils.contour_img(x1[0],h1[0])
            fig.savefig(fpath.format(k)) 
            plt.close()

        fpath = os.getcwd() + "\\Explainers\\"+ self.title + "\\feature_imgs\\"
        spath = os.getcwd() + "\\Explainers\\"+ self.title + "\\explanations\\{}\\".format(name)
        def node_string(fidx,score,weight):
            
            
            nodestr = ""
            nodestr += "<table border=\"0\">\n"
            nodestr+="<tr>"
            nodestr+="<td><img src= \"{}\" /></td>".format(spath+"feature_{}.jpg".format(fidx)) 
            nodestr+="<td><img src= \"{}\" /></td>".format(fpath+"{}.jpg".format(fidx)) 
            nodestr+="</tr>\n"
            if display_value:
                nodestr +="<tr><td colspan=\"2\"><FONT POINT-SIZE=\"{}\"> ClassName: {}, Feature: {}</FONT></td></tr>\n".format(font,classesLoader.No2Name[cidx],fidx)
                nodestr +="<tr><td colspan=\"2\"><FONT POINT-SIZE=\"{}\"> Similarity: {:.3f}, Weight: {:.3f}, Contribution: {:.3f}</FONT></td></tr> \n".format(font,score,weight,score*weight)
            nodestr += "</table>  \n" 
            return nodestr




        s = h.mean(axis = (0,1))
        for i,cidx in enumerate(target_classes):
            tw = w[:,self.C2IDX[cidx]]
            tw_idx = tw.argsort()[::-1][:featuretopk] 
            
            total = 0

            resstr = "digraph Tree {node [shape=plaintext] ;\n"
            resstr += "1 [label=< \n<table border=\"0\"> \n"
            for fidx in tw_idx:
                resstr+="<tr><td>\n"
                    
                resstr+=node_string(fidx,s[fidx],tw[fidx])
                total+=s[fidx]*tw[fidx]
                    
                resstr+="</td></tr>\n"

            if with_total:
                resstr +="<tr><td><FONT POINT-SIZE=\"{}\"> Total Conrtibution: {:.3f}, Prediction: {:.3f}</FONT></td></tr> \n".format(font,total,pred[i])
            resstr += "</table> \n >];\n"
            resstr += "}"

            graph = pydotplus.graph_from_dot_data(resstr)  
            graph.write_jpg(spath+"explanation_{}.jpg".format(cidx))
            graph.write_jpg(afpath+"{}_{}.jpg".format(name,cidx))