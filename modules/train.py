from torch.cuda import is_available
from tqdm import tqdm
import logging
import torch

def train_epoch(model, train_dataloader, test_dataloader, optimizer, loss_fn, device):
    '''Trains a single epoch, helper function for (function) train_model'''
    train_loss = 0
    for input, target in tqdm(train_dataloader):
        input, target = input.to(device), target.to(device)

        # forward pass
        optimizer.zero_grad()
        prediction = model(input)
        loss = loss_fn(prediction, target)

        # backpropagation
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_dataloader)
    
    with torch.no_grad():
        val_loss = 0
        for input, target in tqdm(test_dataloader):
            input, target = input.to(device), target.to(device)
            
            prediction = model(input)
            loss = loss_fn(prediction, target)
            val_loss += loss.item()
        val_loss /= len(test_dataloader) 

    print(f"Train Loss: {train_loss}, Validation Loss: {val_loss}")
    logging.warning(f"Train Loss: {train_loss}, Validation Loss: {val_loss}")
    
    
def train_model(model, train_dataloader, test_dataloader, optimizer, loss_fn, epochs):
    '''Trains the model for all epochs'''
    device = 'cuda' if is_available() else 'cpu'
    for epoch in range(epochs):
        print(f"Epoch: {epoch + 1}")
        logging.warning(f"Epoch: {epoch + 1}")
        train_epoch(model, train_dataloader, test_dataloader, optimizer, loss_fn, device)
        print("------------------------")
        logging.warning("------------------------")
    print("Finished training")
