from torch.cuda import is_available
from tqdm import tqdm

def train_epoch(model, data_loader, optimizer, loss_fn, device):
    print(len(data_loader.dataset))
    for input, target in tqdm(data_loader):
        input, target = input.to(device), target.to(device)

        # forward pass
        prediction = model(input)
        loss = loss_fn(prediction, target)

        # backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Loss: {loss.item()}")

def train_model(model, data_loader, optimizer, loss_fn, epochs):
    device = 'cuda' if is_available() else 'cpu'
    for epoch in range(epochs):
        print(f"Epoch: {epoch}")
        train_epoch(model, data_loader, optimizer, loss_fn, device)
        print("------------------------")
    print("Finished training")