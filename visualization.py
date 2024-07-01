import argparse
import pickle
import matplotlib.pyplot as plt

# get pickle file name
parser = argparse.ArgumentParser()
parser.add_argument('--file_name', type=str, help="pickle file name")
args = parser.parse_args()

file_name = "results/pkls/" + args.file_name

# extract results from files
with open(file_name, 'rb') as file:
    train_loss_list, test_loss_list, accuracy_list = pickle.load(file)

epochs = list(range(1, len(train_loss_list) + 1))

# train loss and test loss visualization
plt.figure(figsize=(5, 5))
plt.plot(epochs, train_loss_list, label='Train Loss')
plt.plot(epochs, test_loss_list, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Train and Test Loss')
plt.legend()
plt.grid(True)
plt.show()

# accuracy visualization
plt.figure(figsize=(5, 5))
plt.plot(epochs, accuracy_list, label='Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy')
plt.legend()
plt.grid(True)
plt.show()