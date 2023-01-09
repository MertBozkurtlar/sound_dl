from modules import dataset, model, train, constants, preprocess
import torch
import os

def train_pipeline():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if not os.path.exists(constants.model_save_loc):
        os.makedirs(constants.model_save_loc)
    
    # Data #
    if not os.path.exists(constants.model_save_loc + "/dataset.pt"):
        # Load the phase (phi) data
        data = preprocess.iterate_all_files(constants)
        # Create the dataset
        data_set = dataset.SpeechDataset(data)
        # Clear the data held on memory
        del data
        # Save dataset
        # torch.save(data_set, constants.model_save_loc + "/dataset.pt")
        # print(f"Loaded data is saved at {constants.model_save_loc}")
    else:
        # If dataset already exist at path, load it instead
        data_set = torch.load(constants.model_save_loc + "/dataset.pt")
        print(f"Loaded data from {constants.model_save_loc}")
    # Feed the data loader
    data_loader = dataset.create_data_loader(data_set, constants.batch_size, shuffle=True)

    # Model #
    print(f"Using {device} device")
    vm_net = model.VonMisesNetwork(size_in=constants.input_size).to(device)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(vm_net.parameters(), lr=constants.learning_rate)

    # Train #
    train.train_model(vm_net, data_loader, optimizer, loss_fn, constants.epochs)

    # Save the model #
    torch.save(vm_net.state_dict(), constants.model_save_loc + "/vm_model.pth")
    print(f"Trained model is saved at {constants.model_save_loc}")

    return vm_net, data_loader
    
    
def test_pipeline():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    data = preprocess.iterate_all_files(constants)
    data_set = dataset.SpeechDataset(data)
    data_loader = dataset.create_data_loader(data_set, 1, shuffle=True)
    # Model #
    print(f"Using {device} device")
    vm_net = model.VonMisesNetwork(size_in=constants.input_size).to(device)
    
    vm_net.load_state_dict(torch.load(constants.model_save_loc + "/vm_model.pth"))
    
    vm_net.eval()
    return vm_net, data_loader

def get_next_pred(model, dataloader):
    inputs, label = next(iter(dataloader))
    inputs = inputs.to("cuda")
    label = label.argmax() * 5
    pred = model(inputs)
    pred = pred.argmax().item() * 5
    print(f"Predicted: {pred} ; Actual: {label}")
