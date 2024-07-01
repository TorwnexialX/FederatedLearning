# Report

> This is a reading report on the paper: [Communication-Efficient Learning of Deep Networks from Decentralized Data](https://arxiv.org/abs/1602.05629)

## Problem It Deals With
The paper introduces federated learning that allows models to be trained on decentralized data located on mobile devices, which saves communication cost and protect personal privacy. The learning process is mainly achieved by algorithm called ***FederatedAveraging***, denoted as ***FedAvg*** in the following report. The algorithm enables models to be trained locally and performs well on real-world data, which is validated with image recognition and text prediction tasks. 

#### Strengths It Has
1. **Privacy Preservation**: Federated learning only updates parameters to the central server and keeps training data on clients, thus protecting data privacy. 
2. **Efficiency in Communication**: The proposed ***FedAvg*** algorithm reduces the communication cost significantly, making it feasible to train deep networks on decentralized data with limited bandwidth.
3. **Robustness to Data Variability**: The algorithm is designed to handle non-IID and unbalanced data, which is closer to real-world scenarios where data distribution across devices varies widely.

#### Method It Provides
***FedAvg*** steps + comparison between 3

##### ***FedAvg*** Steps (Show by PPT illustration)

1. **Initialization**:
   - A global model is initialized on a central server.
2. **Client Selection**:
   - In each round of communication, a subset of clients is randomly selected to participate to conduct local training. 
3. **Local Training**:
   - Each selected client downloads the current global model from the server, and conducts mini-batch SGD training with local datasets in given epochs. 
4. **Model Update Communication**:
   - After local training, each participating client sends its updates(parameters or gradients) back to the central server.
6. **Global Model Update**:
   - The central server updates the global model with parameters collected from clients by averaging.
7. **Iteration**:
   - Steps 2-6 are repeated for multiple rounds until the global model converges to a satisfactory performance level or a predefined stopping criterion is met.

### Comparison

Three kinds of algorithms are mentioned in the paper, including basic optimization algorithm ***SGD***, baseline algorithm under federated setting ***FedSGD***, proposed algorithm ***FedAvg***. 

***SGD***

It is conducted on a single node, and its parameters are updated after every gradient is calculated. 
$$
\theta_{t+1} \leftarrow \theta_t - \eta \nabla L(\theta_t)
$$
where $\eta$ is the learning rate, and $\nabla L(\theta)$ is the gradient of the loss function $L(\theta)$ with respect to the parameters $\theta$, $\theta_t$ marks the value of parameter $\theta$ at time $t$. 

***FedSGD***

It is conducted on all clients in parallel, update parameters to the server after every gradient is calculated. 
$$
\theta_{t+1} \leftarrow \theta_t - \eta \frac{1}{K} \sum_{k=1}^{K} \nabla L_k(\theta_t)\\
$$
where $K$ is the number of clients, and $\nabla L_k(\theta)$ is the gradient from the $k$-th client.

***FedAvg***

It is conducted on selected clients in parallel, update parameters to the server after $E$ epochs of training is performed locally. 
$$
\theta_{t+1} \leftarrow \sum_{k=1}^{K} \frac{n_k}{N} \theta^{(k)}_t
$$
where $\theta^{(k)}$ is the parameters collected from the $k$-th client, $n_k$ is the number of data points on the $k$-th client, and $N$ is the total number of data points across all clients.

## Conclusion It Draws
The paper proposed the concept of *Federated Learning* and corresponding algorithm ***FedAvg***. Such method boast the strength of protecting privacy, saving communication cost, and performing well on non-IID and unbalanced datasets, which means it provides a better way to serve for privacy-sensitive real-world situations. The paper also shows the experiments results, which further confirms strengths mentioned above. 

