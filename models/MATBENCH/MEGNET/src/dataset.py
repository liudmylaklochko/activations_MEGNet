from torch.utils.data import Dataset, DataLoader
import torch
import pickle
import sys

torch.backends.cudnn.enabled=False

def MEGNet_return_model(config, device):
    
    megnet_path = config.get("megnet_path", ".")
    sys.path.insert(0, megnet_path) 
    
    import matgl as matgl

    megnet_loaded = matgl.load_model("MEGNet-MP-2018.6.1-Eform").to(device)
    
    return megnet_loaded

class StructureDataset(Dataset):
    def __init__(self, structures, targets,indices=None):
        self.structures = structures
        self.targets = torch.tensor(targets, dtype=torch.float32)
        self.indices = list(range(len(structures))) if indices is None else indices
        
    def __len__(self):
        return len(self.structures)

    def __getitem__(self, idx):
        return self.indices[idx], self.structures[idx], self.targets[idx]


def collate_fn(batch):
    indices,structures, targets = zip(*batch)
    return list(indices), list(structures), torch.stack(targets)




def returb_dataloader(batch_size,  SEED=42, path_to_data="mpd_eform.pkl", sliced=5000, load_everything=False):
    
    if load_everything == True:
        print ("sorry, we need to send you a link!")
    else:
        with open(path_to_data, "rb") as f:
            data = pickle.load(f)


        structures = data["structures"][:sliced]
        eforms = data["eform_per_atom"][:sliced]
        
    dataset = StructureDataset(structures, eforms)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    return data_loader 


