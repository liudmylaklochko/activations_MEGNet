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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device == torch.device("cuda"): print("USING GPU")

config = json.load(open("configurations/config_MEGNETT.json"))

torchseed = config['seed'] 
pl.seed_everything(torchseed, workers=True)
torch.manual_seed(torchseed)
torch.cuda.manual_seed(torchseed)


print("********  Loading model  ********") 

model = matgl.load_model('MEGNet-MP-2018.6.1-Eform')
model = u.unwrap_model(model)

print("********  Loading dataset  ********")

sliced = 10

with open(config['path_to_train_data'], "rb") as f:
    data = pickle.load(f)

structs = data["structures"]
eforms = data["eform_per_atom"]
mpd_id = data["mp_ids"]


print("********  Getting activations  ********") 

u.activation = {}
list_layers = u.set_up_activations(model,config['framework'])


material_idx = list(range(len(structs)))
shuffle(material_idx)

batch = {}
batch_mp_ids = []

for i, idx in enumerate(material_idx):
    struct = structs[idx]
    u.activation = {}
    if i %100 == 0:
        print(i) 
    with torch.no_grad():
        pred = model.predict_structure(struct)        
    
    structure_added = False

    for layer in u.activation:
        acts = u.get_activations_megnet(u.activation[layer])
        print(acts.size())

        if acts is None or len(acts) <= 1 : continue
        
        if layer not in batch: 
            batch[layer] = []
            structure_added = True
        else: 
            if len(batch[layer]) != 0 and len(acts) != len(batch[layer][0]):
                #print("Bad layer: ", layer)
                batch[layer][0] = []
            else: 
                batch[layer].append(acts)
                structure_added = True
    if structure_added:
        batch_mp_ids.append(mpd_id[idx])

torch.save({'activations': batch, 'mp_ids': batch_mp_ids}, 'activations.pt')