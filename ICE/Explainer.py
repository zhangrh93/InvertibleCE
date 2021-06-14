import sys
sys.path.append(".")

from ICE.utils import *
import ICE.ModelWrapper as ModelWrapper
import ICE.ChannelReducer as ChannelReducer

import os
from pathlib import Path
import pickle

import pydotplus

import numpy as np
import matplotlib.pyplot as plt
import time


FONT_SIZE = 30
#CALC_LIMIT = 3e4
CALC_LIMIT = 1e8
#CALC_LIMIT = 1e9
TRAIN_LIMIT = 50
REDUCER_PATH = "reducer/resnet50"
USE_TRAINED_REDUCER = False
ESTIMATE_NUM = 10

class Explainer():
    def __init__(self,
                 title = "",
                 layer_name = "",
                 class_names = None,
                 utils = None,
                 keep_feature_images = True,
                 useMean = True,
                 reducer_type = "NMF",
                 n_components = 10,
                 featuretopk = 20,
                 featureimgtopk = 5,
                 epsilon = 1e-4):
        self.title = title
        self.layer_name = layer_name
        self.class_names = class_names
        self.class_nos = len(class_names) if class_names is not None else 0

        self.keep_feature_images = keep_feature_images
        self.useMean = useMean
        self.reducer_type = reducer_type
        self.featuretopk = featuretopk
        self.featureimgtopk = featureimgtopk #number of images for a feature
        self.n_components = n_components
        self.epsilon = epsilon
        
        self.utils = utils

        self.reducer = None
        self.feature_distribution = None

        self.feature_base = []
        self.features = {}

        self.exp_location = Path('Explainers')

        self.font = FONT_SIZE
        
    def load(self):
        title = self.title
        with open(self.exp_location / title / (title+".pickle"),"rb") as f:
            tdict = pickle.load(f)
            self.__dict__.update(tdict)
            
    def save(self):
        if not os.path.exists(self.exp_location):
            os.mkdir(self.exp_location)
        title = self.title
        if not os.path.exists(self.exp_location / title):
            os.mkdir(self.exp_location / title)
        with open(self.exp_location / title / (title+".pickle"),"wb") as f:
            pickle.dump(self.__dict__,f)

    def train_model(self,model,loaders):
        self._train_reducer(model,loaders)
        self._estimate_weight(model,loaders)

    def _train_reducer(self,model,loaders):

        print ("Training reducer:")

        if self.reducer is None:
            if not self.reducer_type in ChannelReducer.ALGORITHM_NAMES:
                print ('reducer not exist')
                return 

            if ChannelReducer.ALGORITHM_NAMES[self.reducer_type] == 'decomposition':
                self.reducer = ChannelReducer.ChannelDecompositionReducer(n_components = self.n_components,reduction_alg = self.reducer_type)
            else:
                self.reducer = ChannelReducer.ChannelClusterReducer(n_components = self.n_components,reduction_alg = self.reducer_type)
        
        X_features = []
        for loader in loaders:
            X_features.append(model.get_feature(loader,self.layer_name))
        print ('1/5 Featuer maps gathered.')

        if not self.reducer._is_fit:
            nX_feature = np.concatenate(X_features)
            total = np.product(nX_feature.shape)
            l = nX_feature.shape[0]
            if total > CALC_LIMIT:
                p = CALC_LIMIT / total
                print ("dataset too big, train with {:.2f} instances".format(p))
                idx = np.random.choice(l,int(l*p),replace = False)
                nX_feature = nX_feature[idx]

            print ("loading complete, with size of {}".format(nX_feature.shape))
            start_time = time.time()
            nX = self.reducer.fit_transform(nX_feature)

            print ("2/5 Reducer trained, spent {} s.".format(time.time()-start_time))
        

        self.cavs = self.reducer._reducer.components_
        nX = nX.mean(axis = (1,2))
        self.feature_distribution = {"overall":[(nX[:,i].mean(),nX[:,i].std(),nX[:,i].min(),nX[:,i].max()) for i in range(self.n_components)]}

        reX = []
        self.feature_distribution['classes'] = []
        for X_feature in X_features:
            t_feature = self.reducer.transform(X_feature)
            pred_feature = self._feature_filter(t_feature)
            self.feature_distribution['classes'].append([pred_feature.mean(axis=0),pred_feature.std(axis=0),pred_feature.min(axis=0),pred_feature.max(axis=0)])
            reX.append(self.reducer.inverse_transform(t_feature))

        err = []
        for i in range(len(self.class_names)):
            res_true = model.feature_predict(X_features[i],layer_name=self.layer_name)[:,i] #
            res_recon = model.feature_predict(reX[i],layer_name=self.layer_name)[:,i] #
            err.append(abs(res_true-res_recon).mean(axis=0) / (abs(res_true.mean(axis=0))+self.epsilon))


        self.reducer_err = np.array(err)
        if type(self.reducer_err) is not np.ndarray:
            self.reducer_err = np.array([self.reducer_err])

        print ("3/5 Error estimated, fidelity: {}.".format(self.reducer_err))

        return self.reducer_err

    def _estimate_weight(self,model,loaders):
        if self.reducer is None:
            return

        X_features = []

        for loader in loaders:
            X_features.append(model.get_feature(loader,self.layer_name)[:ESTIMATE_NUM])
        X_feature = np.concatenate(X_features)

        print ('4/5 Weight estimator initialized.')
        
        self.test_weight = []
        for i in range(self.n_components):
            cav = self.cavs[i,:]

            res1 =  model.feature_predict(X_feature - self.epsilon * cav,layer_name=self.layer_name)
            res2 =  model.feature_predict(X_feature + self.epsilon * cav,layer_name=self.layer_name)

            res_dif = res2 - res1
            dif = res_dif.mean(axis=0) / (2 * self.epsilon)
            if type(dif) is not np.ndarray:
                dif = np.array([dif])
            self.test_weight.append(dif)
        
        print ('5/5 Weight estimated.')

        self.test_weight = np.array(self.test_weight)

    def generate_features(self,model,loaders):
        self._visulize_features(model,loaders)
        self._save_features()
        if self.keep_feature_images == False:
            self.features = {}
        return

    def _feature_filter(self,featureMaps,threshold = None):
    #filter feature map to feature value with threshold for target value
        if self.useMean:
            res = featureMaps.mean(axis = (1,2))
        else:
            res = featureMaps.max(axis=(1,2))
        if threshold is not None:
            res = - abs(res - threshold) 
        return res
   
    def _update_feature_dict(self,x,h,nx,nh,threshold = None):

        if type(x) == type(None):
            return nx,nh
        else:
            x = np.concatenate([x,nx])
            h = np.concatenate([h,nh])

            nidx = self._feature_filter(h,threshold = threshold).argsort()[-self.featureimgtopk:]
            x = x[nidx,...]
            h = h[nidx,...]
            return x,h

    def _visulize_features(self,model,loaders, featureIdx = None, inter_dict = None):
        featuretopk = min(self.featuretopk, self.n_components)

        imgTopk = self.featureimgtopk
        if featureIdx is None:
            featureIdx = []
            tidx = []
            w = self.test_weight
            for i,_ in enumerate(self.class_names):
                tw = w[:,i]
                tidx += tw.argsort()[::-1][:featuretopk].tolist()
            featureIdx += list(set(tidx))                    

        nowIdx = set(self.features.keys())
        featureIdx = list(set(featureIdx) - nowIdx)
        featureIdx.sort()

        if len(featureIdx) == 0:
            print ("All feature gathered")
            return

        print ("visulizing features:")
        print (featureIdx)

        features = {}
        for No in featureIdx:
            features[No] = [None,None]
            
        if inter_dict is not None:
            for k in inter_dict.keys():
                inter_dict[k] = [[None,None] for No in featureIdx]
        
        print ("loading training data")
        for i,loader in enumerate(loaders):
            
            for X in loader:
                X = X[0]
                featureMaps = self.reducer.transform(model.get_feature(X,self.layer_name))
                
                X_feature = self._feature_filter(featureMaps)

                for No in featureIdx:
                    samples,heatmap = features[No]
                    idx = X_feature[:,No].argsort()[-imgTopk:]
                    
                    nheatmap = featureMaps[idx,:,:,No]
                    nsamples = X[idx,...]
                    
                    samples,heatmap = self._update_feature_dict(samples,heatmap,nsamples,nheatmap)
                    
                    features[No] = [samples,heatmap]
                    
                    if inter_dict is not None:
                        for k in inter_dict.keys():
                            vmin = self.feature_distribution['overall'][No][2]
                            vmax = self.feature_distribution['overall'][No][3]
                            temp_v = (vmax - vmin) * k + vmin
                            inter_dict[k][No] = self._update_feature_dict(inter_dict[k][No][0],inter_dict[k][No][1],X,featureMaps[:,:,:,No],threshold = temp_v)
                        
                    
            print ("Done with class: {}, {}/{}".format(self.class_names[i],i+1,len(loaders)))
        # create repeat prototypes in case lack of samples
        for no,(x,h) in features.items():
            idx = h.mean(axis = (1,2)).argmax()
            for i in range(h.shape[0]):
                if h[i].max() == 0:
                    x[i] = x[idx]
                    h[i] = h[idx]
        
        self.features.update(features)
        self.save()
        return inter_dict

    def _save_features(self,threshold=0.5,background = 0.2,smooth = True):
        feature_path = self.exp_location / self.title / "feature_imgs"
        #utils = self.utils

        if not os.path.exists(feature_path):
            os.mkdir(feature_path)

        for idx in self.features.keys(): 

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
                timg = self.utils.deprocessing(x[i])
                if timg.max()>1:
                    timg = timg / 255.0
                    timg = abs(timg)
                timg = np.clip(timg,0,1)
                nimg[:,i*self.utils.img_size[1]:(i+1)*self.utils.img_size[1],:] = timg
                nh[:,i*self.utils.img_size[1]:(i+1)*self.utils.img_size[1]] = h[i]
            fig = self.utils.contour_img(nimg,nh)
            fig.savefig(feature_path / (str(idx)+".jpg"),bbox_inches='tight',pad_inches=0)
            plt.close(fig)

    def global_explanations(self):        
        title = self.title
        fpath = (self.exp_location / self.title / "feature_imgs").absolute()
        feature_topk = min(self.featuretopk,self.n_components)
        feature_weight = self.test_weight
        class_names = self.class_names
        Nos = range(self.class_nos)

        font = self.font
        
        def LR_graph(wlist,No):
            def node_string(count,fidx,w,No):
                nodestr = ""
                nodestr += "{} [label=< <table border=\"0\">".format(count)

                nodestr+="<tr>"
                nodestr+="<td><img src= \"{}\" /></td>".format(str(fpath / ("{}.jpg".format(fidx)))) 
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
                count-=1
            
            resstr += "0 [label=< <table border=\"0\">" 
            resstr += "<tr><td><FONT POINT-SIZE=\"{}\"> ClassName: {} </FONT></td></tr>" .format(font,class_names[No])
            resstr += "<tr><td><FONT POINT-SIZE=\"{}\"> Fidelity error: {:.3f} % </FONT></td></tr>" .format(font,self.reducer_err[No]*100)
            resstr += "<tr><td><FONT POINT-SIZE=\"{}\"> First {} features out of {} </FONT></td></tr>" .format(font,feature_topk,self.n_components)
            resstr += "</table>  >];\n"
            

            resstr += "}"

            return resstr

        if not os.path.exists(self.exp_location / title / "GE"):
            os.mkdir(self.exp_location / title / "GE")
                    
        print ("Generate explanations with fullset condition")

        for i in Nos:
            wlist = [(j,feature_weight[j][i]) for j in feature_weight[:,i].argsort()[-feature_topk:]]
            graph = pydotplus.graph_from_dot_data(LR_graph(wlist,i))  
            graph.write_jpg(str(self.exp_location / title / "GE" / ("{}.jpg".format(class_names[i]))))
 
    def local_explanations(self,x,model,background = 0.2,name = None,with_total = True,display_value = True):
        utils = self.utils
        font = self.font
        featuretopk = min(self.featuretopk,self.n_components)

        target_classes = list(range(self.class_nos))
        w = self.test_weight

        pred = model.predict(x)[0][target_classes]
        
        fpath = self.exp_location / self.title / "explanations"

        if not os.path.exists(fpath):
            os.mkdir(fpath)

        afpath = fpath / "all"

        if not os.path.exists(afpath):
            os.mkdir(afpath)


        if name is not None:
            fpath = fpath / name
            if not os.path.exists(fpath):
                os.mkdir(fpath)
            else:
                print ("Folder exists")
                return 
        else:
            count = 0
            while os.path.exists(fpath / str(count)):
                count+=1
            fpath = fpath / str(count)
            os.mkdir(fpath)
            name = str(count)


        if self.reducer is not None:
            h = self.reducer.transform(model.get_feature(x,self.layer_name))[0]
        else:
            h = model.get_feature(x,self.layer_name)[0]

        feature_idx = []
        for cidx in target_classes:
            tw = w[:,cidx]
            tw_idx = tw.argsort()[::-1][:featuretopk]
            feature_idx.append(tw_idx)
        feature_idx = list(set(np.concatenate(feature_idx).tolist()))

        

        for k in feature_idx:
            
            minmax = False
            if self.reducer_type == "PCA":
                minmax = True

            x1,h1 = utils.img_filter(x,np.array([h[:,:,k]]),background=background,minmax = minmax)
            x1 = utils.deprocessing(x1)
            x1 = x1 / x1.max()
            x1 = abs(x1)
            fig = utils.contour_img(x1[0],h1[0])
            fig.savefig(fpath / ("feature_{}.jpg".format(k)))
            plt.close()

        fpath = fpath.absolute()
        gpath = self.exp_location.absolute() / self.title / 'feature_imgs'
        def node_string(fidx,score,weight):
            
            
            nodestr = ""
            nodestr += "<table border=\"0\">\n"
            nodestr+="<tr>"
            nodestr+="<td><img src= \"{}\" /></td>".format(str(fpath / ("feature_{}.jpg".format(fidx))))
            nodestr+="<td><img src= \"{}\" /></td>".format(str(gpath / ("{}.jpg".format(fidx))))
            nodestr+="</tr>\n"
            if display_value:
                nodestr +="<tr><td colspan=\"2\"><FONT POINT-SIZE=\"{}\"> ClassName: {}, Feature: {}</FONT></td></tr>\n".format(font,self.class_names[cidx],fidx)
                nodestr +="<tr><td colspan=\"2\"><FONT POINT-SIZE=\"{}\"> Similarity: {:.3f}, Weight: {:.3f}, Contribution: {:.3f}</FONT></td></tr> \n".format(font,score,weight,score*weight)
            nodestr += "</table>  \n" 
            return nodestr




        s = h.mean(axis = (0,1))
        for cidx in target_classes:
            tw = w[:,cidx]
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
                resstr +="<tr><td><FONT POINT-SIZE=\"{}\"> Total Conrtibution: {:.3f}, Prediction: {:.3f}</FONT></td></tr> \n".format(font,total,pred[cidx])
            resstr += "</table> \n >];\n"
            resstr += "}"

            graph = pydotplus.graph_from_dot_data(resstr)  
            graph.write_jpg(str(fpath / ("explanation_{}.jpg".format(cidx))))
            graph.write_jpg(str(afpath / ("{}_{}.jpg".format(name,cidx))))