import torch
import matplotlib.pyplot as plt    
import numpy as np
import sys,os, json
import importlib as imp
import pickle 


def unwrap_model(model):
    # Common patterns for wrapped models
    if hasattr(model, "model") and isinstance(model.model, torch.nn.Module):
        return model.model
    return model

def get_activations_megnet(t):
    if type(t) == tuple and len(t) > 1: t = t[0] # only the first one for now...
    if type(t) == tuple and len(t) > 1: t = t[0] # can be a tuple in a tuple
    if len(t.shape) == 1: return t.detach()
    if len(t.shape) == 2:
        if t.shape[0] == 1: return t[0].detach()
        else: return torch.mean(t, dim=1) # could be other aggregation methods
    if len(t.shape) == 3:
        if t.shape[0] == 1 and t.shape[1] == 1: return t[0][0].detach()
        else: return torch.mean(torch.mean(t, dim=2), dim=1).detach() # randomly, I don't think this happens
    # print("!!!!!!!!!!!!", len(t.shape))
    return None


def set_up_activations(model,framework):
    if framework == 'PyTorch':
        global activation
        llayers = []
        def get_activation(name):
            def hook(model, input, output):
                if type(output) != torch.Tensor: activation[name] = output
                else: 
                    activation[name] = output.cpu().detach()
                    # print(name, activation[name].shape)
            return hook

        def rec_reg_hook(mo, prev="", lev=0):
            for k in mo.__dict__["_modules"]:
                name = prev+"."+k if prev != "" else k
                nmo = getattr(mo,k)
                nmo.register_forward_hook(get_activation(name))
                #print("--"+"--"*lev, "hook added for",name)
                llayers.append(name)
                rec_reg_hook(nmo, prev=name, lev=lev+1)
            return llayers
        return rec_reg_hook(model)
    elif framework == 'TensorFlow':
        from tensorflow.keras.models import Model
        layer_names=None
        keras_model = model.model
    if layer_names is None:
        layer_names = [layer.name for layer in keras_model.layers if 'input' not in layer.name]

    intermediate_model = Model(inputs=keras_model.input,outputs=[keras_model.get_layer(name).output for name in layer_names])
    return intermediate_model, layer_names


