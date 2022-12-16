from modules import dataset, model, train, constants
import torch

def train_pipeline():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Data
    data = dataset.SpeechDataset(constants)
    data_loader = dataset.create_data_loader(data, constants.batch_size)

    # Model
    print(f"Using {device} device")
    vm_net = model.Von_Mises_Network(size_in=constants.input_size).to(device)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(vm_net.parameters(), lr=constants.learning_rate)

    # Train
    train.train_model(vm_net, data_loader, optimizer, loss_fn, constants.epochs)

    # Save the model
    torch.save(vm_net.state_dict(), constants.model_save_loc)
    print(f"Trained model is saved at {constants.model_save_loc}")