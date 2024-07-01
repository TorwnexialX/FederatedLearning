from torch import nn
import torch.nn.functional as F

# The model structure follows the `2NN` architecture described in the paper, fixing most of the parameters.
# However, to achieve higher accuracy in limited rounds, a dropout layer is included.
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