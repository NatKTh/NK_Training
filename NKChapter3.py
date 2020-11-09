import torch
import torchvision
from torchvision import datasets , transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchsummary import summary
import os

TestWhat = 2
batch = 32


if TestWhat == 1: # ImageFolder
    nkclassification = 2 # Cat and Dog
    dstransform = transforms.Compose([
        transforms.Resize((64,64)),
        transforms.RandomRotation((-5,5)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,0.5,0.5),std=(0.25,0.25,0.25))
    ])

    trainds1 = datasets.ImageFolder("./train",dstransform)
    validds1 = datasets.ImageFolder("./val",dstransform)
    #testds1 = datasets.ImageFolder("./test",dstransform)

    traindl1 = DataLoader(trainds1,batch,shuffle = True,pin_memory=True)
    validdl1 = DataLoader(validds1,batch,shuffle = True,pin_memory=True)
    #testdl = DataLoader(testds,batch,shuffle = True,pin_memory=True)

elif TestWhat == 2: # MNIST datasets
    nkclassification = 10 # Number 0 - 9
    mnisttransform = transforms.Compose([
        transforms.Resize((28,28)),
        transforms.Grayscale(),
        transforms.RandomRotation(degrees=(-5,5)),
        transforms.ToTensor()
    ])

    pathdb = "./datamnist"
    trainds2 = datasets.MNIST(pathdb ,transform = mnisttransform, train = True , download = True)
    validds2 = datasets.MNIST(pathdb ,transform = mnisttransform, train = False , download = True)

    traindl2 = DataLoader(trainds2,batch,shuffle = True,pin_memory=True)
    validdl2 = DataLoader(validds2,batch,shuffle = True,pin_memory=True)


class NKCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(NKCNN, self).__init__()
        if TestWhat == 1:
            self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        elif TestWhat == 2:
            self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x    

if torch.cuda.is_available():
    torch.cuda.empty_cache()
    device = torch.device("cuda")
    print("Using GPU : ",torch.cuda.get_device_name(0))
    #print(torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print("Using CPU")

if TestWhat == 1:
    NK_Checkpoint = "./tmp/checkpoint.pkl"
    result_file = os.path.isfile('./tmp/checkpoint.pkl')
elif TestWhat == 2:
    NK_Checkpoint = "./tmp/checkpoint2.pkl"
    result_file = os.path.isfile('./tmp/checkpoint2.pkl')

nk = NKCNN(nkclassification)
if result_file == True:
    nk_state_dict = torch.load(NK_Checkpoint)
    nk.load_state_dict(nk_state_dict)
    print("Load last model parameters")
optimizer = optim.Adam(nk.parameters(), lr=0.001)
nk.to(device)

def check_cuda_using():
    if device.type == 'cuda':
        #print(torch.cuda.get_device_name(0))
        #print('Memory Usage:')
        #print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
        #print('Cached:   ', round(torch.cuda.memory_cached(0)/1024**3,1), 'GB')
        print('Memory Usage:Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB''Cached:   ', round(torch.cuda.memory_cached(0)/1024**3,1), 'GB')
def train(model, optimizer, loss_fn, train_loader, val_loader, epochs=20, device="cpu"):
    print("Start Running (Please Wait!)")
    for epoch in range(epochs):
        training_loss = 0.0
        valid_loss = 0.0
        #check_cuda_using()
        model.train()
        #print("Start Training : epoch =  " , epoch , "Dataloader length = ",len(train_loader))
        for batch in train_loader:
            optimizer.zero_grad()
            inputs, targets = batch
            inputs = inputs.to(device)
            targets = targets.to(device)
            output = model(inputs)
            loss = loss_fn(output, targets)
            loss.backward()
            optimizer.step()
            training_loss += loss.data.item() * inputs.size(0)
        training_loss /= len(train_loader.dataset)
        
        model.eval()
        #print("Start Evaluation : epoch = " , epoch, "Dataloader length = ",len(val_loader))
        num_correct = 0 
        num_examples = 0
        for batch in val_loader:
            inputs, targets = batch
            inputs = inputs.to(device)
            output = model(inputs)
            targets = targets.to(device)
            loss = loss_fn(output,targets) 
            valid_loss += loss.data.item() * inputs.size(0)
            correct = torch.eq(torch.max(F.softmax(output), dim=1)[1], targets).view(-1)
            num_correct += torch.sum(correct).item()
            num_examples += correct.shape[0]
        valid_loss /= len(val_loader.dataset)

        print('Epoch: {}, Training Loss: {:.2f}, Validation Loss: {:.2f}, accuracy = {:.2f} %'.format(epoch, training_loss,
        valid_loss, num_correct / num_examples*100))

if TestWhat == 1:
    train(nk, optimizer,torch.nn.CrossEntropyLoss(), traindl1,validdl1, epochs=10, device=device)
elif TestWhat == 2:
    train(nk, optimizer,torch.nn.CrossEntropyLoss(), traindl2,validdl2, epochs=2, device=device)
    
torch.save(nk.state_dict(),NK_Checkpoint)
print("Finish Computation!")
if TestWhat == 1:
    summary(nk,input_size=(3,64,64))
elif TestWhat == 2:
    summary(nk,input_size=(1,28,28))
   