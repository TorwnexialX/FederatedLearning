from tqdm import tqdm
import random
from utils import *
from models import *
from settings import *

args = args_parser()

# initialize dataset
idxs, train_dataset, test_dataset = get_dataset(if_iid=args.if_iid, num_clients=args.K)

# initialize global model
global_model = MNIST_2NN()
global_model.to(args.device)
global_model.train()

# initialize lists of results
train_loss_list = []
test_loss_list = []
accuracy_list = []

with tqdm(range(args.rounds)) as global_bar:
    global_bar.colour = "blue"

    # global training
    for epoch in global_bar:
        global_bar.set_description(f"Global Round {epoch + 1}")

        # select clients subset
        m = int(max(args.C * args.K, 1))
        S = random.sample(range(args.K), m)

        # lists to store local parameters and losses
        local_state_dict = []
        local_loss = []

        with tqdm(total=len(S)) as local_bar:
            local_bar.colour = "red"
            local_bar.leave = False
            local_bar.set_description("Local Training")
            for k in S:
                # ClientUpdate
                client = Client(global_model, args)
                client_state_dict, client_loss = client.update(train_dataset, idxs, k)
                local_state_dict.append(client_state_dict)
                local_loss.append(client_loss)
                local_bar.update(1)
            local_bar.close()

        # param average
        avg_state_dict = param_average(local_state_dict)
        global_model.load_state_dict(avg_state_dict)

        # evaluation
        train_loss = np.mean(local_loss)
        test_loss, accuracy = evaluate(global_model, test_dataset, args.device)
        printout = \
            f'Global Round: {epoch + 1}\n'\
            f'accuracy: {accuracy * 100}% \n'\
            f'train_loss: {train_loss} \n'\
            f'test_loss: {test_loss}'
        global_bar.write(printout)
    