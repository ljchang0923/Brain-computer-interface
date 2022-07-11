import torch.nn as nn
import torch
from torch.nn import init

class basic_CNN(nn.Module):
    def __init__(self, filter_size):
        super(basic_CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 50, (1, filter_size), padding=(0, 25), bias=False),
            nn.BatchNorm2d(50)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(50, 50, (22, 1), groups=50, bias=False),
            nn.BatchNorm2d(50),
            nn.ELU(),
            nn.MaxPool2d((1, 3), stride=3),
            # nn.AvgPool2d((1, 62), stride=(1, 12),
            nn.Dropout(0.5)
        )
    def forward(self, input):
        x = self.conv1(input)
        x = self.conv2(x) #shape of x:(batch,50,1,196)
        return x

class CNN1(nn.Module):
    def __init__(self, h1=5000):
        super(CNN1, self).__init__()
        self.basic_cnn = basic_CNN(30)
        self.fc = nn.Linear(h1,1024) # sliding:6500 / 1050 normal:9700 
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(1024, 4)

    def forward(self, input):
        x = self.basic_cnn(input)
        # print(f"latent size: {x.size()}")
        x = torch.flatten(x,1)
        latent = self.fc(x)
        x = self.relu(latent)
        x = self.dropout(x)
        output = self.classifier(x)

        return latent, output

class CNN2(nn.Module):
    def __init__(self,h1=3400):
        super(CNN2, self).__init__()
        self.basic_cnn = basic_CNN(25)
        self.conv_block = nn.Sequential(
            nn.Conv2d(50, 100, (1,10), padding=(0,5), bias=False),
            nn.BatchNorm2d(100),
            nn.ELU(),
            nn.MaxPool2d((1,3), stride=3),
            nn.Dropout(0.5)
        )
        self.fc = nn.Linear(h1,1024) # sliding: 3400 normal:6500
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(1024, 4)

    def forward(self, input):
        x = self.basic_cnn(input)
        x = self.conv_block(x)
        # print(f"latent size: {x.size()}")
        x = torch.flatten(x,1)
        latent = self.fc(x)
        x = self.relu(latent)
        x = self.dropout(x)
        output = self.classifier(x)

        return latent, output

class CNN3(nn.Module):
    def __init__(self, h1=1200):
        super(CNN3, self).__init__()
        self.basic_cnn = basic_CNN(20)
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(50, 100, (1,10), padding=(0,5), bias=False),
            nn.BatchNorm2d(100),
            nn.ELU(),
            nn.MaxPool2d((1,3), stride=3),
            nn.Dropout(0.5)
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(100, 100, (1,10), padding=(0,5), bias=False),
            nn.BatchNorm2d(100),
            nn.ELU(),
            nn.MaxPool2d((1,3), stride=3),
            nn.Dropout(0.5)
        )
        self.fc = nn.Linear(h1, 1024) # sliding:1200 normail:2200
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(1024, 4)

    def forward(self, input):
        x = self.basic_cnn(input)
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        # print(f"latent size: {x.size()}")
        x = torch.flatten(x,1)
        latent = self.fc(x)
        x = self.relu(latent)
        x = self.dropout(x)
        output = self.classifier(x)
        return latent, output

class CNN4(nn.Module):
    def __init__(self, h1=400):
        super(CNN4, self).__init__()
        self.basic_cnn = basic_CNN(10)
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(50, 100, (1,10), padding=(0,5), bias=False),
            nn.BatchNorm2d(100),
            nn.ELU(),
            nn.MaxPool2d((1,3), stride=3),
            nn.Dropout(0.5)
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(100, 100, (1,10), padding=(0,5), bias=False),
            nn.BatchNorm2d(100),
            nn.ELU(),
            nn.MaxPool2d((1,3), stride=3),
            nn.Dropout(0.5)
        )
        self.fc = nn.Linear(h1, 1024)  #sliding: 400 normal:700
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(1024, 4)
        
    def forward(self, input):
        x = self.basic_cnn(input)
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block2(x)
        # print(f"latent size: {x.size()}")
        x = torch.flatten(x,1)
        latent = self.fc(x)
        x = self.relu(latent)
        x = self.dropout(x)
        output = self.classifier(x)
        return latent, output

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(4096,50)
        self.drop1 = nn.Dropout(0.5)
        self.elu = nn.ELU()
        self.fc2 = nn.Linear(50,50)
        self.drop2 = nn.Dropout(0.5)
        #self.softmax = nn.LogSoftmax(dim=1)
        self.classifier = nn.Linear(50,4)

        self.layers = nn.Sequential(self.fc1,self.drop1,self.elu,self.fc2,self.drop2,self.classifier)
        self.init_weights()

    def init_weights(self):
        init.xavier_uniform_(self.fc1.weight)
        init.xavier_uniform_(self.fc2.weight)

    def forward(self, input):
        '''x = self.fc1(input)
        x = self.fc2(x)
        output = self.classifier(x)'''
        output = self.layers(input)
        dummy = 0
        return dummy, output

class AE(nn.Module):
    def __init__(self):
        super(AE, self).__init__()
        self.encode = nn.Linear(4096,100)
        self.decode = nn.Linear(100,4096)
        self.classifier = nn.Linear(4096,4)

    def forward(self, input):
        x = self.encode(input)
        output = self.decode(x)
        with torch.no_grad():
            pred = self.classifier(output)

        return output, pred

class EEGNet(nn.Module):
    def __init__(self):
        super(EEGNet, self).__init__()

        self.F1 = 8
        self.F2 = 16
        self.D = 2

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, self.F1, (1, 64), padding=(0, 32), bias=False),
            nn.BatchNorm2d(self.F1)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(self.F1, self.D*self.F1, (22, 1), groups=self.F1, bias=False),
            nn.BatchNorm2d(self.D*self.F1),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(0.5)
        )

        self.Conv3 = nn.Sequential(
            nn.Conv2d(self.D*self.F1, self.D*self.F1, (1, 16), padding=(0, 8), groups=self.D*self.F1, bias=False),
            nn.Conv2d(self.D*self.F1, self.F2, (1, 1), bias=False),
            nn.BatchNorm2d(self.F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(0.5)
        )

        self.classifier = nn.Linear(16*17, 4, bias=True)
        #self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.Conv3(x)
        
        x = x.view(-1, 16*17)
        x = self.classifier(x)
        #x = self.softmax(x)
        return 0,x

class ShallowConvNet(nn.Module):
    def __init__(self):
        super(ShallowConvNet, self).__init__()

        self.conv1 = nn.Conv2d(1, 40, (1, 13), bias=False)
        self.conv2 = nn.Conv2d(40, 40, (22, 1), bias=False)
        self.Bn1   = nn.BatchNorm2d(40)
        # self.SquareLayer = square_layer()
        self.AvgPool1 = nn.AvgPool2d((1, 35), stride=(1, 7))
        # self.LogLayer = Log_layer()
        self.Drop1 = nn.Dropout(0.25)
        self.classifier = nn.Linear(40*74, 4, bias=True)
        #self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.Bn1(x)
        x = x ** 2
        x = self.AvgPool1(x)
        x = torch.log(x)
        x = self.Drop1(x)
        x = x.view(-1, 40*74)
        x = self.classifier(x)

        #x = self.softmax(x)
        return 0,x

class SCCNet(nn.Module):
    def __init__(self):
        super(SCCNet, self).__init__()
        # bs, 1, channel, sample
        self.conv1 = nn.Conv2d(1, 22, (22, 1))
        self.Bn1 = nn.BatchNorm2d(22)
        # bs, 22, 1, sample
        self.conv2 = nn.Conv2d(22, 20, (1, 12), padding=(0, 6))
        self.Bn2   = nn.BatchNorm2d(20)
        # self.SquareLayer = square_layer()
        self.Drop1 = nn.Dropout(0.5)
        self.AvgPool1 = nn.AvgPool2d((1, 62), stride=(1, 12))
        self.classifier = nn.Linear(840, 4, bias=True)
        #self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.conv1(x)
        x = self.Bn1(x)
        x = self.conv2(x)
        x = self.Bn2(x)
        x = x ** 2
        x = self.Drop1(x)
        x = self.AvgPool1(x)
        x = torch.log(x)
        x = x.view(-1, 840)
        x = self.classifier(x)

        #x = self.softmax(x)
        return 0,x