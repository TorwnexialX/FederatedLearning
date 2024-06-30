import argparse

def args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--rounds', type=int, default=1000,
                        help="number of coummunication rounds")
    parser.add_argument('--E', type=int, default=10,
                        help="number of local epochs")
    parser.add_argument('--B', type=int, default=10,
                        help="the local minibatch size")
    parser.add_argument('--K', dest='K', const=100, default=100, action='store_const', 
                        help="number of clients (constant: 100)")
    parser.add_argument('--C', type=float, default=0.1,
                        help="the fraction of clients that perform computation on each round")
    parser.add_argument('--lr', type=float, default=0.01,
                        help="learning rate")
    
    parser.add_argument('--if_iid', type=bool, default=False,
                        help="whther training set will be I.I.D.")
    parser.add_argument('--device', type=str, default="cuda", choices=["cuda", "cpu"], 
                        help="conduct the training on \"cuda \" or \"cpu\"")
    
    args = parser.parse_args()
    return args
