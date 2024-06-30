import pickle
import matplotlib.pyplot as plt

# extract results from files
with open('results.pkl', 'rb') as file:
    train_loss_list, test_loss_list, accuracy_list = pickle.load(file)

# train loss and test loss visualization
plt.figure(figsize=(10, 5))
plt.plot(train_loss_list, label='Train Loss')
plt.plot(test_loss_list, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Train and Test Loss')
plt.legend()
plt.grid(True)
plt.show()

# accuracy visualization
plt.figure(figsize=(10, 5))
plt.plot(accuracy_list, label='Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy')
plt.legend()
plt.grid(True)
plt.show()