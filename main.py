from modules import dataset, model, train, constants, preprocess
import torch
import os

def train_pipeline():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Data
    data = preprocess.iterate_all_files(constants)
    data_set = dataset.SpeechDataset(data)
    data_loader = dataset.create_data_loader(data_set, constants.batch_size, shuffle=True)

    # Model
    print(f"Using {device} device")
    vm_net = model.Von_Mises_Network(size_in=constants.input_size).to(device)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(vm_net.parameters(), lr=constants.learning_rate)

    # Train
    train.train_model(vm_net, data_loader, optimizer, loss_fn, constants.epochs)

    # Save the model
    if not os.path.exists(constants.model_save_loc):
        os.makedirs(constants.model_save_loc)
    torch.save(vm_net.state_dict(), constants.model_save_loc + "/vm_model.pth")
    print(f"Trained model is saved at {constants.model_save_loc}")
    
    
def test_pipeline():
    pass