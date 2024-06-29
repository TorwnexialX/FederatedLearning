from torch import nn
import torch.nn.functional as F

class MNIST_2NN(nn.Module):
    def __init__(self):
        super(MNIST_2NN, self).__init__()
        input_units  = 28 * 28
        hidden_units = 200
        out_units    = 10
        self.layer_input = nn.Linear(input_units, hidden_units) # 156,800 + 200
        self.relu = nn.ReLU()
        self.hidden1 = nn.Linear(hidden_units, hidden_units) # 40,000 + 200
        self.hidden2 = nn.Linear(hidden_units, out_units) # 2,000 + 10
        self.dropout = nn.Dropout()

    def forward(self, x):
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3]) # batch_size * flat
        x = self.layer_input(x)
        x = self.hidden1(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.hidden2(x)
        return x

class CNNMnist(nn.Module):
    def __init__(self, args):
        super(CNNMnist, self).__init__()
        self.conv1 = nn.Conv2d(args.num_channels, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, args.num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)