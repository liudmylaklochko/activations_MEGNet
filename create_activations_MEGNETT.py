#%%
import sys
project_root = "models/MATBENCH/MEGNET/src/"
sys.path.append(project_root)

import torch, sys, json
import importlib as imp
import utils_f as u
import models.MATBENCH.MEGNET.src.matgl as matgl
import pickle
from random import shuffle
import lightning as pl
from collections import defaultdict
torch.backends.cudnn.enabled=False
from tqdm import tqdm

def is_effectively_empty(value):
    # Check if it's a list that contains only empty lists or is empty itself
    return isinstance(value, list) and all(
        (isinstance(item, list) and len(item) == 0) for item in value
    )

def run_acivations(calculate_act = True):

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


    def unwrap_model(model):
        # Common patterns for wrapped models
        if hasattr(model, "model") and isinstance(model.model, torch.nn.Module):
            return model.model
        return model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == torch.device("cuda"): print("USING GPU")

    config = json.load(open("configurations/config_MEGNETT.json"))

    torchseed = config['seed'] 
    pl.seed_everything(torchseed, workers=True)
    torch.manual_seed(torchseed)
    torch.cuda.manual_seed(torchseed)


    print("********  Loading model  ********") 

    model = matgl.load_model('MEGNet-MP-2018.6.1-Eform')
    model = unwrap_model(model)

    print("********  Loading dataset  ********")

    sliced = 10

    with open(config['path_to_train_data'], "rb") as f:
        data = pickle.load(f)

    structures = data["structures"][:sliced]
    eforms = data["eform_per_atom"][:sliced]
    mpd_id = data["mp_ids"][:sliced]


    print("********  Getting activations  ********") 

    u.activation = {}
    list_layers = u.set_up_activations(model,config['framework'])

    shuffle(structures)

    batch = {}
    shuffle(structures)

    call_back_global = defaultdict(list)
    call_back = {}
    layer_acts = {} 
    count = 0 
    if calculate_act == True:
        for i, struct in enumerate(structures):
            u.activation = {}
            with torch.no_grad():
                pred = model.predict_structure(struct)
            for layer in u.activation: 
                if type(u.activation[layer]) == tuple: u.activation[layer] = u.activation[layer][0]
                acts = u.activation[layer].to(device)
                if config["aggregation"] == "flatten": 
                    if u.activation[layer].dim() == 1: 
                        print("Bad layer: ", layer)
                        continue
                    else:
                        acts = torch.flatten(u.activation[layer], start_dim=1).to(device)
                elif config["aggregation"] == "mean":
                    if len(u.activation[layer].shape) > 2:
                        acts = torch.mean(u.activation[layer], dim=1).to(device)
                    else: acts = u.activation[layer].to(device)
                call_back_global[layer].append(acts.cpu())   

            #for layer in u.activation:
            #    acts = get_activations_megnet(u.activation[layer])  
            #    if acts is None or len(acts) <= 1 : continue
            #    if layer not in batch: batch[layer] = []
            #    else: 
            #        if len(batch[layer]) != 0 and len(acts) != len(batch[layer][0]):
                        # print("Bad layer: ", layer)
            #            batch[layer][0] = []
            #        else: 
            #            batch[layer].append(acts)
                           
            count += 1
        call_back = {layer: acts for layer, acts in call_back_global.items() if layer in u.activation}        
        print("********  Storing activations  ********")

        for layer in call_back.keys():
            filename = f"acts_{layer}.pt"                        
            try:
                full_acts = torch.cat(call_back[layer], dim=0) 
                layer_acts[layer] = full_acts            
                
                torch.save(full_acts, f"{config['actdir']}/{filename}")
                #print(f"Saved {filename} at {config['actdir']}")
            except Exception as e:
                print(f"Error saving {filename}: {e}")
    
    if calculate_act == True:
        data_back = u.return_tuple(layer_acts,structures, model,list(call_back.keys()),count)
    else:
        data_back = u.return_tuple(layer_acts,structures, model,list_layers,count)
    
    if not data_back:
        raise RuntimeError("return_tuple returned an empty tuple or None. Aborting cleanup.")
    
    del model, structures, layer_acts 
    
    return data_back