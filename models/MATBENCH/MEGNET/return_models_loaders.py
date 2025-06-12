import torch, sys, json
import warnings
import lightning as pl
from dgl.data.utils import split_dataset
import pickle 

# To suppress warnings for clearer output
warnings.simplefilter("ignore")

def MEGNet_return_dataset(config, batch,  SEED, path_to_data="mp_structures_dataset.pkl", sliced=5000, load_everything=False):
    
    megnet_path = config.get("megnet_path", ".")
    sys.path.insert(0, megnet_path) 
    from MATBENCH.MEGNET.src.matgl.ext.pymatgen import Structure2Graph, get_element_list
    from MATBENCH.MEGNET.src.matgl.graph.data import MGLDataLoader, MGLDataset, collate_fn_graph
    
    with open(path_to_data, "rb") as f:
        data = pickle.load(f)


    structures = data["structures"][:sliced]
    eforms = data["eform_per_atom"][:sliced]
    
    # get element types in the dataset
    elem_list = get_element_list(structures)
    
    # setup a graph converter

    converter = Structure2Graph(element_types=elem_list, cutoff=4.0)

    # convert the raw dataset into MEGNetDataset

    mp_dataset = MGLDataset(
    structures=structures,
    labels={"Eform": eforms},
    converter=converter)

    train_data, val_data, test_data = split_dataset(
    mp_dataset,
    frac_list=[0.8, 0.1, 0.1],
    shuffle=True,
    random_state=SEED)

    train_loader, val_loader, test_loader = MGLDataLoader(
    train_data=train_data,
    val_data=val_data,
    test_data=test_data,
    collate_fn=collate_fn_graph,
    batch_size=batch,
    num_workers=0)
    
    return val_loader
    
def MEGNet_return_model(config, device):
    
    
    megnet_path = config.get("megnet_path", ".")
    sys.path.insert(0, megnet_path) 
    import MATBENCH.MEGNET.src.matgl as matgl 
    from MATBENCH.MEGNET.src.matgl.utils.training import ModelLightningModule
    
    
    
       
    torchseed = 42 
    pl.seed_everything(torchseed, workers=True)
    torch.manual_seed(torchseed)
    torch.cuda.manual_seed(torchseed)

    max_epochs = 300



    #megnet_loaded = matgl.load_model(config['model_path']).to(device)
    megnet_loaded = matgl.load_model('MEGNet-MP-2018.6.1-Eform').to(device)
    lit_module = ModelLightningModule(model=megnet_loaded)
     
    
    trainer = pl.Trainer(max_epochs=max_epochs, accelerator='cpu')
    # trainer.test(lit_module, dataloaders=val_loader)
    
    return lit_module, trainer
