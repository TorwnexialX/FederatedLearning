import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import torch
from models import MNIST_2NN
import copy

def get_dataset(if_iid:bool, num_clients):
    """
    Given the distribution type, then returns the dataset. 

    Args:
        if_iid: train_dataset will be distributed I.I.D when True, not when False.
    
    Returns:
        idxs: indices of examples distributed to each client
        train_dataset: MNIST training dataset. 
        test_dataset: MNIST test dataset. 
    """
    data_dir = 'data/'
    num_clients = int(num_clients)

    # Introduce negative numbers through normalization, 
    # which has been experimentally proven to improve accuracy in a limited number of rounds.
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5), (0.5))])

    train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    test_dataset  = datasets.MNIST(data_dir, train=False, download=True, transform=transform)

    if if_iid == True:
        # shuffle 60,000 training images
        idxs = np.arange(len(train_dataset))
        np.random.shuffle(idxs)

        # distribute shuffled images to 100 clients evenly
        idxs = idxs.reshape([num_clients, -1])

    if if_iid == False:
        # distribute 60,000 images to 300 shards, each 200 images
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
        self.global_model = global_model  # global model from server
        self.lr = args.lr                 # learning rate
        self.B = args.B                   # batch size
        self.E = args.E                   # local epochs
        self.device = args.device         # device

    def update(self, global_trainset, idxs, k):
        """
        update the model parameters locally

        Args:
            global_trainset: the entire training set
            idxs: indices of examples distributed to each client
            k: index of current client
        
        Returns:
            locally updated weight and final loss
        """
        model = MNIST_2NN().to(self.device)
        criterion = torch.nn.CrossEntropyLoss()
        trainset = Subset(global_trainset, idxs[k])
        trainloader = DataLoader(trainset, batch_size=self.B, shuffle=True)
        
        model.load_state_dict(self.global_model.state_dict())
        # optimizer customization
        optimizer = torch.optim.SGD(model.parameters(), lr=self.lr, momentum=0.5)
        model.train()
        for epoch in range(self.E):
            for features, labels in trainloader:
                features, labels = features.to(self.device), labels.to(self.device)
                model.zero_grad()
                outputs = model(features)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

        return model.state_dict(), loss.item()

def param_average(dict_list):
    """
    Given state_dicts of clients, then averages the parameters. 

    Args:
        dict_list: list of state_dicts from clients

    Returns:
        new_dict: the state_dict whose parameters are the averaged ones
    """
    # Initialize the new dictionary with the same structure as the input dictionaries
    new_dict = copy.deepcopy(dict_list[0])
    
    for key in new_dict.keys():
        # calculate the mean of each key by stacking the key values and then taking the average
        stacked_tensors = torch.stack([d[key] for d in dict_list], dim=0)
        mean_tensor = torch.mean(stacked_tensors, dim=0)
        new_dict[key] = mean_tensor
    
    return new_dict

def evaluate(global_model, global_testset, device):
    """
    Evaluate the global model on the entire test set. 

    Args:
        global_model: the global model
        global_testset: the entire test set
        device: the device (CPU or GPU) to run the evaluation on
    
    Returns:
        avg_loss: average loss
        accuracy: accuracy over the test set
    """
    testloader = DataLoader(global_testset, batch_size=64, shuffle=False)
    global_model.eval()
    
    # initialize loss function, total loss, and correct prediction counter
    criterion = torch.nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for features, labels in testloader:
            features, labels = features.to(device), labels.to(device)
            outputs = global_model(features)

            # add the test loss
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            # count the correct prediction in the current batch
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    
    # calculate average loss and accuracy
    avg_loss = total_loss / len(testloader)
    accuracy = correct / total
    
    return avg_loss, accuracy