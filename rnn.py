import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda
import matplotlib.pyplot as plt
import os
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import DataLoader
from pathlib import Path
import ConvLSTM
import torch.optim as optim

class CustomImageDataset(Dataset):
    def __init__(self, image_directory): 
        self.image_directory = image_directory

        self.total_num_inputs = 0

        for dir_name in Path(image_directory).glob('*'):
            self.total_num_inputs += len(list(dir_name.glob('*.pt')))

        self.tensor_labels = ["norm", "mi"]

    def __len__(self):
        return self.total_num_inputs

    def __getitem__(self, idx):
        #there are 7547 norm images and 5486 mi images
        #return the correct label and the corresponding tensor (loaded by the file!)

        if idx <= 7546:
            label = self.tensor_labels[0]
            tensor_file_path = "./tensorfiles_rnn/norm/tensor" + str(idx) + ".pt"
            tensor = torch.load(tensor_file_path) 
        else:
            label = self.tensor_labels[1]
            tensor_file_path = "./tensorfiles_rnn/mi/tensor" + str(idx-7547) + ".pt"
            tensor = torch.load(tensor_file_path)

        return (tensor.squeeze(), torch.tensor(0.0 if label == self.tensor_labels[0] else 1.0))

training_data = CustomImageDataset("./tensorfiles_rnn")

train_dataloader = DataLoader(training_data, batch_size=2, shuffle=True)

# Display image and label.
train_features, train_labels = next(iter(train_dataloader))

print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")

plt.imshow(train_features[0].squeeze()[0].squeeze().permute(1, 2, 0))



class FlatConvLSTM(torch.nn.Module):
    """An ConvLSTM layer that ignores the current hidden and cell states."""
    def __init__(self):
        super().__init__()
        self.convlstm = ConvLSTM.ConvLSTM(3, 10, (3,3), 1, True, True, False)

    def forward(self, x):
        _, lstm_output = self.convlstm(x)
        return lstm_output[0][0]


model = torch.nn.Sequential(
    FlatConvLSTM(),
    torch.nn.Flatten(),
    torch.nn.Linear(10*480*640, 1),
    torch.nn.Sigmoid()
)


#create the loss function
loss = torch.nn.BCELoss()
optimizer = optim.Adam(model.parameters())
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

num_epochs = 10
model.to(device)

for epoch in range(num_epochs):
    
    # Set model to training mode
    model.train()
    
    # Update the model for each batch
    train_count = 0
    train_cost = 0
    batch = 0

    for X, y in train_dataloader:
        
        # Compute model cost
        #yhat = model(X.view(-1, nx))
        X = X.to(device)
        y = y.to(device)

        yhat = model(X)
        #print(yhat.shape, y.shape)

        try:
            cost = loss(yhat.squeeze(), y)
            model.zero_grad()
            cost.backward()
        except:
            print()

        
        train_count += X.shape[0]
        train_cost += cost.item()
        optimizer.step()
        print(epoch, batch, cost.item())
        batch += 1

    # Set model to evaluation mode
    model.eval()
    
    # Test model on validation data
    valid_count = 0
    valid_cost = 0
    valid_correct = 0
            
    train_cost /= train_count
    
    print(epoch, train_cost)

print('Done.')

#save the model in a file
torch.save(model, "./rnn_saved_models/model.py")
torch.save(model.state_dict(), "./rnn_saved_models/model_parameters.py")
