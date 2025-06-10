#%%
import sys
project_root = "models/MATBENCH/megnet_master"
sys.path.append(project_root)

import torch, json
import utils_f as u
import pickle 
from models.MATBENCH.megnet_master.megnet.models import MEGNetModel  
from datasets.dataset import StructureDataset 
import tensorflow as  tf
from collections import defaultdict
from tqdm import tqdm

def run_acivations(calculate_act = True):


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == torch.device("cuda"): print("USING GPU")

    config = json.load(open("configurations/config_megnet.json"))
    torch.manual_seed(config['seed'])

    print("********  Loading model  ********")   

    model = MEGNetModel.from_file(config['model'])
    #model = model.to(device)

    print("********  Loading dataset  ********")

    with open(config['path_to_train_data'], "rb") as f:
            data = pickle.load(f)


    dataset = StructureDataset(structures=data["structures"][:10], targets=data['eform_per_atom'][:10])

    graphs, targets = dataset.return_graph_megnet(model)

    batch_size = config["batchsize"]
    falt_graph = model.graph_converter.get_flat_data(graphs)
    inputs =  model._create_generator(*falt_graph, batch_size=batch_size, is_shuffle=False)


    ## for a single graph : inputs = model.graph_converter.graph_to_input(graphs_valid[0]) 


    print("********  Getting activations  ********")   

    u.activation = {}
    activation_model,list_layers = u.set_up_activations(model,framework=config['framework'])
    print(list_layers)
    
    call_back = {}
    layer_acts = {} 
    call_back_global = defaultdict(list)
    count = 0 
    if calculate_act == True : 
        for x in tqdm(inputs, desc="Calculating activations"):
            count = count + 1 
            u.activation = {}
            outputs = activation_model(x)
            u.activation = {name: output for name, output in zip(list_layers, outputs)}
            
            for layer in list_layers:
                if type(u.activation[layer]) == tuple or type(u.activation[layer]) == list : 
                    u.activation[layer] = u.activation[layer][0]
                u.activation[layer] = torch.from_numpy(u.activation[layer].numpy())
                #u.activation[layer] = tf.convert_to_tensor(u.activation[layer])
                if config["aggregation"] == "flatten": 
                    acts = torch.flatten(u.activation[layer], start_dim=1).to(device)
                elif config["aggregation"] == "mean":
                    if len(u.activation[layer].shape) > 2:
                        acts = torch.mean(u.activation[layer], dim=1).to(device)
                    else: acts = u.activation[layer].to(device)
                else: 
                    print("unknown aggregation, check config")
                    sys.exit(-1)
                call_back_global[layer].append(acts.cpu())   
        call_back = {layer: acts for layer, acts in call_back_global.items() if layer in u.activation}        
        print("********  Storing activations  ********")
        for layer in u.activation:                        
            try:
                full_acts = torch.cat(call_back[layer], dim=1) 
                layer_acts[layer] = full_acts            
                filename = f"acts_{layer}.pt"
                torch.save(full_acts, f"{config['actdir']}/{filename}")
                #print(f"Saved {filename} at {config['actdir']}")
            except Exception as e:
                print(f"Error saving {filename}: {e}")
    
    if calculate_act == True:
        data_back = u.return_tuple(layer_acts,inputs, model,list(u.activation.keys()),count)
    else:
        data_back = u.return_tuple(layer_acts,inputs, model,list_layers, count)
    
    if not data_back:
        raise RuntimeError("return_tuple returned an empty tuple or None. Aborting cleanup.")
    
    del model, inputs    
    
    
    return data_back        


# tuple to return : 
# data_back = u.return_tuple(call_back,inputs)