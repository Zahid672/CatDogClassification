from torch import nn



# __all__ = ['CNN_model'] ## list of all the functions, classes, variables to import from this file



class CNN_model(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CNN_model, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(in_features=64*28*28, out_features=2048) ### fully connected layer 1
        self.fc2 = nn.Linear(in_features=2048, out_features=512)
        self.fc3 = nn.Linear(in_features=512, out_features=128)
        self.out = nn.Linear(in_features=128, out_features=out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.25)

    def forward(self, x):
        x = self.conv1(x) #[N, C, H, W, .. ]
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv3(x)
        x = self.relu(x)
        x = self.pool(x)

        x = x.view(-1, 64*28*28) ## flatten the output of the convolution layers

        x = self.relu(self.fc1(x))  #[N, C*H*W*...]
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.dropout(x)

        x = self.out(x) ### fully connected layer 2

        return x