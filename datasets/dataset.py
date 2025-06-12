from torch.utils.data import Dataset, DataLoader
import torch
import pickle


class StructureDataset(Dataset):
    def __init__(self, structures, targets, indices=None):
        self.structures = structures
        self.targets = torch.tensor(targets, dtype=torch.float32)
        self.indices = list(range(len(structures))) if indices is None else indices
        
    def __len__(self):
        return len(self.structures)

    def __getitem__(self, idx):
        return self.indices[idx], self.structures[idx], self.targets[idx],  self.activation_functions[idx]

    def return_graph_megnet(self, model):
        graphs_valid = []
        targets_valid = []
        structures_invalid = []
        for s, p in zip(self.structures, self.targets):
            try:
                graph = model.graph_converter.convert(s)
                graphs_valid.append(graph)
                targets_valid.append(p)
            except:
                structures_invalid.append(s)
        return  graphs_valid, targets_valid  



def collate_fn(batch):
    indices,structures, targets = zip(*batch)
    return list(indices), list(structures), torch.stack(targets)

    
