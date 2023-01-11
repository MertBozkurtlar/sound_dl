from torch.cuda import is_available
from torch.utils.data import Dataset, DataLoader
import torch
from modules import constants


class SpeechDataset(Dataset):
    def __init__(self, data):
        super().__init__()
        self.device = 'cuda' if is_available() else 'cpu'
        self.data = data
        self.degree_step = constants.degree_step

    def __len__(self):
        return len(self.data[1])

    def __getitem__(self, index):
        spec, label = self.data
        spec = torch.from_numpy(spec[index])
        label = self.encode_label(label[index])
        spec.to(self.device)
        label.to(self.device)
        return spec, label
    
    # Encode the label in one-hot vector 
    def encode_label(self, label):
        label = int(label / self.degree_step)
        vector = torch.zeros(int(360 / self.degree_step))
        vector[label] = 1
        return vector


def create_data_loader(train_data, batch_size, shuffle):
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=shuffle)
    return train_dataloader


# # Data #
#     if not os.path.exists(constants.model_save_loc + "/dataset.pt"):
#         # Load the phase (phi) data
#         data = preprocess.iterate_all_files(constants)
#         # Create the dataset
#         data_set = dataset.SpeechDataset(data)
#         # Clear the data held on memory
#         del data
#         # Save dataset
#         # torch.save(data_set, constants.model_save_loc + "/dataset.pt")
#         # print(f"Loaded data is saved at {constants.model_save_loc}")
#     else:
#         # If dataset already exist at path, load it instead
#         data_set = torch.load(constants.model_save_loc + "/dataset.pt")
#         print(f"Loaded data from {constants.model_save_loc}")
#     # Feed the data loader