'''
@Author: Zhang Ruihan
@Date: 2019-10-28 01:01:52
@LastEditors  : Zhang Ruihan
@LastEditTime : 2020-01-07 01:33:49
@Description: file content
'''

import numpy as np
import os
import torch
from keras import backend as K
from ClassesLoader import *

class ModelWrapper:
    def __init__(self,model,batch_size = 128):
        self.model = model
        self.batch_size = batch_size
    def get_feature(self,x,layer_name):
        pass
    def feature_predict(self,feature,layer_name = None):
        pass  
    def predict(self,x):
        pass
    def target_value(self,x):
        pass
    
class KerasModelWrapper(ModelWrapper):
    def __init__(self,
                 model,
                 layer_dict = {},
                 target = None,
                 channel_last = False,
                 input_size = [224, 224, 3],
                 batch_size=128):#target: (layer_name,unit_nums)
        self.layer_dict = layer_dict
        self.target = target
        self.channel_last = channel_last
        self.input_size = list(input_size)

        super().__init__(model,batch_size)
        
    def _batch_fn(self,x,fn):
        batch_size = self.batch_size
        l = x.shape[0]
        it_num = l // batch_size
        res = []
        fr = 0
        to = 0
        for i in range(it_num):
            fr = i*batch_size
            to = (i+1)*batch_size
            res.append(fn([x[fr:to]])[0])
        res.append(fn([x[to:]])[0])
        res = np.concatenate(res)
        return res
            
    def get_feature(self,x,layer_name = None):
        if layer_name == None:
            return self.model.predict(x)
        output = self.model.get_layer(layer_name).output
        fn = K.function([self.model.input],[output])
        
        features = self._batch_fn(x,fn)
        return features

    def feature_predict(self,feature,layer_name = None): 
        if layer_name == None:
            return self.model.predict(feature)
        output = self.model.get_layer(layer_name).output
        fo = K.function([output],[self.model.output])
        
        pred = self._batch_fn(feature,fo)
        return pred

    def target_predict(self,feature,layer_name = None):        

        if self.target is None:
            print ("No target")
            return None
        if layer_name =="input":
            finput = self.model.input
        else:
            finput = self.model.get_layer(layer_name).output

        target_layer,unit = self.target

        if target_layer == "output":
            output = self.model.layers[-1].input
            w,b = self.model.layers[-1].get_weights()

            fo = K.function([finput],[output])
            
            pred = self._batch_fn(feature,fo)

            pred = np.dot(pred,w)+b

        else:
            output = self.model.get_layer(target_layer).output
            fo = K.function([finput],[output])
            
            pred = self._batch_fn(feature,fo)

        return pred[...,unit]
    
    def predict(self,x):
        return self.model.predict(x)


class Keras2ModelWrapper(ModelWrapper):
    def __init__(self,
                 model,
                 layer_dict = {},
                 target = None,
                 channel_last = False,
                 input_size = [224, 224, 3],
                 batch_size=128):#target: (layer_name,unit_nums)
        self.layer_dict = layer_dict
        self.target = target
        self.channel_last = channel_last
        self.input_size = list(input_size)

        
        super().__init__(model,batch_size)
        
    def _batch_fn(self,x,fn):
        batch_size = self.batch_size
        l = x.shape[0]
        it_num = l // batch_size
        res = []
        fr = 0
        to = 0
        for i in range(it_num):
            fr = i*batch_size
            to = (i+1)*batch_size
            res.append(fn([x[fr:to]])[0])
        res.append(fn([x[to:]])[0])
        res = np.concatenate(res)
        return res
            
    def get_feature(self,x,layer_name = None):
        if layer_name == None:
            return self.model.predict(x)
        output = self.model.get_layer(layer_name).output
        fn = K.function([self.model.input],[output])
        
        features = self._batch_fn(x,fn)
        return features

    def feature_predict(self,feature,layer_name= None): 
        layer_name,unit = self.target
        if layer_name == None:
            return self.model.predict(feature)
        output = self.model.get_layer(layer_name).output
        fo = K.function([output],[self.model.output])
        
        pred = self._batch_fn(feature,fo)
        return pred

    def target_predict(self,feature,layer_name = None):        

        if self.target is None:
            print ("No target")
            return None
        if layer_name =="input":
            finput = self.model.input
        else:
            finput = self.model.get_layer(layer_name).output

        target_layer,unit = self.target

        output = self.model.get_layer(target_layer).output
        fo = K.function([finput],[output])
        
        pred = self._batch_fn(feature,fo)

        return pred[...,unit]
    
    def predict(self,x):
        return self.model.predict(x)


class PytorchModelWrapper(ModelWrapper):   
    def __init__(self,
                 model,
                 layer_dict = {},
                 target = None,
                 channel_last = False,
                 input_size = [3,224,224],
                 batch_size=128):#target: (layer_name,unit_nums)
        self.layer_dict = layer_dict
        self.layer_dict.update(dict(model.named_children()))
        self.target = target
        self.channel_last = channel_last
        self.input_size = list(input_size)
        
        self.CUDA = torch.cuda.is_available()

        super().__init__(model,batch_size)

    def set_target(self,targetNos = None, target_layer = None):
        self.target = (target_layer,targetNos)

    def _to_tensor(self,x):
        if type(x) == np.ndarray:
            x = torch.from_numpy(x)
        return x

    def _switch_channel_f(self,x):
        if not self.channel_last:
            if x.ndim == 3:
                x = np.transpose(x,(2,0,1))
            else:
                x = np.transpose(x,(0,3,1,2))
        return x

    def _switch_channel_b(self,x):
        if not self.channel_last:
            if x.ndim == 3:
                x = np.transpose(x,(1,2,0))
            elif x.ndim == 4:
                x = np.transpose(x,(0,2,3,1))
        return x

    def _fun(self,x,layer_in = "input",layer_out = "output"):
        #cpu in cpu out

        x = x.type(torch.FloatTensor)


        in_flag = False
        if layer_in == "input":
            in_flag = True
        

        data_in = x.clone()
        if self.CUDA:
            data_in = data_in.cuda()
        data_out = []

        handles = []
        
        def hook_in(m,i,o):
            return data_in
        def hook_out(m,i,o):
            data_out.append(o)

        if layer_in == "input":
            nx = x
        else:
            handles.append(self.layer_dict[layer_in].register_forward_hook(hook_in))
            nx = torch.zeros([x.size()[0]]+self.input_size)

        if not layer_out == "output":
            handles.append(self.layer_dict[layer_out].register_forward_hook(hook_out))

        if self.CUDA:
            nx = nx.cuda()
            
        with torch.no_grad():
            #print (nx.sum())
            ny = self.model(nx)

        #print(data_out)

        if layer_out == "output":
            data_out = ny
        else:
            data_out = data_out[0]

        data_out = data_out.cpu()

        for handle in handles:
            handle.remove() 

        return data_out

    def _batch_fn(self,x,layer_in = "input",layer_out = "output"):
        #tensor in tensor out
        out = []

        batch_size = self.batch_size
        l = x.shape[0]
        it_num = l // batch_size
        fr = 0
        to = 0
        for i in range(it_num):
            fr = i*batch_size
            to = (i+1)*batch_size
            nx = x[fr:to]
            out.append(self._fun(nx,layer_in,layer_out))
        nx = x[to:]
        if nx.shape[0] > 0:
            out.append(self._fun(nx,layer_in,layer_out))

        res = torch.cat(out,0)

        return res

    
    def get_feature(self,x,layer_name):
        x = self._switch_channel_f(x)
        if layer_name not in self.layer_dict:
            return None

        nx = self._to_tensor(x)
        out = self._batch_fn(nx,layer_out = layer_name)
        out = out.numpy()

        out = self._switch_channel_b(out)
        return out

    def feature_predict(self,feature,layer_name = None):
        feature = self._switch_channel_f(feature)
        if layer_name not in self.layer_dict:
            return None

        nx = self._to_tensor(feature)
        out = self._batch_fn(nx,layer_in = layer_name)
        out = out.numpy()

        out = self._switch_channel_b(out)
        return out        


    def target_predict(self,feature,layer_name = None):
        feature = self._switch_channel_f(feature)

        if self.target is None:
            print ("No target")
            return None
        if layer_name not in self.layer_dict:
            print ("layer not found")
            return None

        target_layer,unit_nums = self.target
        nx = self._to_tensor(feature)
        out = self._batch_fn(nx,layer_in = layer_name,layer_out = target_layer)
        out = out.numpy()

        out = self._switch_channel_b(out)
        return out[...,unit_nums]


    def predict(self,x):
        x = self._switch_channel_f(x)

        nx = self._to_tensor(x)
        out = self._batch_fn(nx)
        out = out.numpy()
        
        out = self._switch_channel_b(out)
        return out