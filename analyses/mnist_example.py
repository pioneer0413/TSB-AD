import torch
from torch import nn, optim
from torchvision import datasets, transforms
import wandb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def get_mnist_data(batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    train_dataset = datasets.MNIST(root='/home/hwkang/dev-TSB-AD/TSB-AD/analyses', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='/home/hwkang/dev-TSB-AD/TSB-AD/analyses', train=False, download=True, transform=transform)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader
def build_mnist_model():
    model = nn.Sequential(
        nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Flatten(),
        nn.Linear(64 * 7 * 7, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    )
    
    return model.to(device)

if __name__ == "__main__":
    train_loader, test_loader = get_mnist_data(batch_size=64)
    model = build_mnist_model()
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    wandb.init(
        project="MNIST-Example",
        name='run'
    )
    
    wandb.watch(model, log="all", log_freq=1)
    # Training loop
    for epoch in range(1, 50):  # 5 epochs for demonstration
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item()}')
        wandb.log({"train_loss": loss.item() / len(train_loader)})
    wandb.finish()

    # Testing loop
    with torch.no_grad():
        model.eval()
        total_correct = 0
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            
            pred = output.argmax(dim=1, keepdim=True)
            total_correct += pred.eq(target.view_as(pred)).sum().item()

    accuracy = total_correct / len(test_loader.dataset)
    print(f'Test Accuracy: {accuracy * 100:.2f}%')