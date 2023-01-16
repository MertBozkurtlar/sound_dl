from torch.cuda import is_available
from tqdm import tqdm
import logging

def train_epoch(model, data_loader, optimizer, loss_fn, device):
    '''Trains a single epoch, helper function for (function) train_model'''
    for input, target in tqdm(data_loader):
        input, target = input.to(device), target.to(device)

        # forward pass
        optimizer.zero_grad()
        prediction = model(input)
        loss = loss_fn(prediction, target)

        # backpropagation
        loss.backward()
        optimizer.step()
    print(f"Loss: {loss.item()}")
    logging.warning(f"Loss: {loss.item()}")

def train_model(model, data_loader, optimizer, loss_fn, epochs):
    '''Trains the model for all epochs'''
    device = 'cuda' if is_available() else 'cpu'
    for epoch in range(epochs):
        print(f"Epoch: {epoch}")
        logging.warning(f"Epoch: {epoch}")
        train_epoch(model, data_loader, optimizer, loss_fn, device)
        print("------------------------")
        logging.warning("------------------------")
    print("Finished training")
