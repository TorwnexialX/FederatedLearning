import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import torch
from models import MNIST_2NN
import copy

def get_dataset(if_iid:bool):
    """
    :param if_iid: train_dataset will be distributed I.I.D when True, not when False.
    :return idxs: indices of examples distributed to each client
    :return train_dataset: MNIST training dataset. 
    :return test_dataset: MNIST test dataset. 
    """
    data_dir = '/data/'
    num_clients = 100

    train_dataset = datasets.MNIST(data_dir, train=True, download=True,
                                    transform=transforms.ToTensor())

    test_dataset  = datasets.MNIST(data_dir, train=False, download=True,
                                    transform=transforms.ToTensor())

    if if_iid == True:
        # shuffle 60,000 training images
        idxs = np.arange(len(train_dataset))
        np.random.shuffle(idxs)

        # distribute shuffled images to 100 clients evenly
        idxs = idxs.reshape([num_clients, -1])

    if if_iid == False:
        # 60,000 training imgs -->  200 imgs/shard X 300 shards
        num_shards, num_imgs = 200, 300
        idxs = np.arange(num_shards*num_imgs)
        labels = train_dataset.train_labels.numpy()

        # sort labels
        idxs_labels = np.vstack((idxs, labels))
        idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
        idxs = idxs_labels[0, :]

        # shuffle shards and distribute them to 100 clients each 2
        idxs = idxs.reshape([num_shards, num_imgs])
        np.random.shuffle(idxs)
        idxs = idxs.reshape([num_clients, -1])

    return idxs, train_dataset, test_dataset

class Client:
    def __init__(self, global_model, args):
        self.global_model = global_model
        self.lr = args.lr
        self.B = args.B
        self.E = args.E
        self.device = args.device

    def update(self, global_trainset, idxs, k):
        """
        :param global_trainset: the entire training set
        :param idxs: indices of examples distributed to each client
        :param k: index of current client
        :return: locally updated weight and final loss
        """
        model = MNIST_2NN().to(self.device)
        criterion = torch.nn.CrossEntropyLoss()
        trainset = Subset(global_trainset, idxs[k])
        trainloader = DataLoader(trainset, batch_size=self.B, shuffle=True)
        batch_num = len(trainloader)
        
        model.load_state_dict(self.global_model.state_dict())
        # optimizer customization
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=1e-4)
        model.train()
        epoch_loss = []
        for epoch in range(self.E):
            total_loss = 0.0
            for features, labels in trainloader:
                features, labels = features.to(self.device), labels.to(self.device)
                model.zero_grad()
                outputs = model(features)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() # loss.item() is the averaged loss of each minibatch
            epoch_loss.append(total_loss / batch_num)

        return model.state_dict(), epoch_loss[-1]

def split_state_dict(state_dict):
    """
    :param state_dict: state_dict of a model
    :return weights_tensor: 2D-tensor of weights in each layer
    :return biases_tensor: 2D-tensor of biases in each layer
    """
    params = list(state_dict.values())
    weights_list = []
    biases_list = []
    for i, value in enumerate(params):
        if i % 2 == 0:
            weights_list.append(value)
        else:
            biases_list.append(value)
    weights_tensor = torch.stack(weights_list)
    biases_tensor = torch.stack(biases_list)
    return weights_tensor, biases_tensor

# def param_average(dict_list):
#     new_dict = copy.deepcopy(dict_list[0])
#     list_len = len(dict_list)
#     for key in new_dict.keys():
#         value = torch.zeros(new_dict[key].shape, device=new_dict[key].device)
#         for i in range(list_len):
#             value += dict_list[i][key]
#         value /= list_len
#         new_dict[key] = value
#     return new_dict

def param_average(dict_list):
    # Initialize the new dictionary with the same structure as the input dictionaries
    new_dict = copy.deepcopy(dict_list[0])
    list_len = len(dict_list)
    
    # Iterate over each key in the dictionary
    for key in new_dict.keys():
        # Stack all tensors along a new dimension and calculate the mean
        stacked_tensors = torch.stack([d[key] for d in dict_list], dim=0)
        mean_tensor = torch.mean(stacked_tensors, dim=0)
        
        # Assign the mean tensor to the new dictionary
        new_dict[key] = mean_tensor
    
    return new_dict

def load_params(global_dict, avg_weights, avg_biases):
    """
    :param global_dict: state_dict() of global model
    :avg_weights: averaged weights np.array
    :avg_biases: averaged biases np.array
    """
    for i, key in enumerate(global_dict.keys()):
        if i % 2 == 0:
            global_dict[key] = avg_weights[int(i/2)]
        else:
            global_dict[key] = avg_biases[int(i/2)]

def evaluate(global_model, global_testset, device):
    """
    :param global_model: the global model
    :param global_testset: the entire test set
    :param device: the device (CPU or GPU) to run the evaluation on
    :return avg_loss: average loss
    :return accuracy: accuracy over the test set
    """
    criterion = torch.nn.CrossEntropyLoss()
    testloader = DataLoader(global_testset, batch_size=64, shuffle=False)
    
    global_model.eval()
    
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for features, labels in testloader:
            features, labels = features.to(device), labels.to(device)
            outputs = global_model(features)

            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    avg_loss = total_loss / len(testloader)
    accuracy = correct / total
    
    return avg_loss, accuracy