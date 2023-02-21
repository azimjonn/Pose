import torch
from torch import nn
from torch.utils.data import DataLoader

from sklearn.model_selection import train_test_split

from data import PoseDataset

BATCH_SIZE = 128
LR = 0.0005
EPOCHS = 10000

dataset = PoseDataset('poses.txt')

train_data, val_data = train_test_split(dataset, test_size=0.2)

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)

print(len(train_loader), len(val_loader))

encoder = nn.Sequential(
    nn.Linear(87, 150),
    nn.ReLU(),
    nn.Linear(150, 150),
    nn.ReLU(),
    nn.Linear(150, 90),
    nn.ReLU(),
    nn.Linear(90, 50)
)

decoder = nn.Sequential(
    nn.Linear(50, 90),
    nn.ReLU(),
    nn.Linear(90, 150),
    nn.ReLU(),
    nn.Linear(150, 150),
    nn.ReLU(),
    nn.Linear(150, 87),
)

class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
    
    def forward(self, x):
        return self.decoder(self.encoder(x))

model = EncoderDecoder(encoder, decoder)
criterion = nn.MSELoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=LR)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

for epoch in range(EPOCHS):
    total_loss = 0
    model.train()
    for iter, x in enumerate(train_loader):
        optimizer.zero_grad()

        x_hat = model.forward(x)
        loss = criterion(x_hat, x)
        total_loss += loss.detach()
        loss.backward()

        optimizer.step()

    print(f'epoch: {epoch}, loss {total_loss/(len(train_loader) * BATCH_SIZE)}')

    if epoch % 1000 == 0:
        torch.save(model.state_dict(), 'model.pth')
        total_loss = 0
        model.eval()
        for iter, x in enumerate(val_loader):
            
            with torch.no_grad():
                x_hat = model.forward(x)
                total_loss += criterion(x_hat, x)

        print(f'val loss {total_loss/(len(val_loader) * BATCH_SIZE)}')