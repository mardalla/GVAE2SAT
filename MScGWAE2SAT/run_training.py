import os
from argparse import ArgumentParser

import torch

from data import get_train_val_dataloader, get_max_shape
from model import get_vgae
from train import train

def main():
    parser = ArgumentParser()
    parser.add_argument("--src", dest="source_dir", type=str,
                        default="../cnfs/cnfs_train",
                        help="path of directory with CNFs to train on")
    parser.add_argument("--dest", dest="dest", type=str, default="models/",
                        help="directory to save trained model in")
    parser.add_argument("--batch_size", dest="batch_size", type=int, default=16,
                        help="batch size for DataLoader")
    parser.add_argument("--latent_size", dest="latent_size", type=int, default=32,
                        help="latent size of VGAE")
    parser.add_argument("--enc_layer_size", dest="enc_size", type=int, default=128,
                        help="size of hidden layers within encoder")
    parser.add_argument("--dec_layer_size", dest="dec_size", type=int, default=256,
                        help="size of hidden layers within decoder")
    parser.add_argument("--num_gnn", dest="num_gnn", type=int, default=2,
                        help="number of graph convolutional layers in encoder")
    parser.add_argument("--num_expansions", dest="num_expansions", type=int,
                        help="number of dense layers in decoder", default=3)
    parser.add_argument("--dropout", dest="dropout", type=float, default=.4,
                        help="dropout rate in decoder")
    parser.add_argument("--lr", dest="lr", type=float, default=.003,
                        help="learning rate of optimiser")
    parser.add_argument("--wd", dest="wd", type=float, default=0.1,
                        help="weight decay for optimiser")
    parser.add_argument("--epoch_num", dest="epoch_num", type=int, default=1,
                        help="number of epochs to train for")
    parser.add_argument("--pres", dest="pres", type=int, default=4,
                        help="multiplicative weight for penalising false negatives")
    parser.add_argument("--beta", dest="beta", type=float, default=.001,
                        help="coefficient of VAE's KL-loss")
    args = parser.parse_args()
    
    max_shape = get_max_shape(args.source_dir)
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    torch.manual_seed(args.batch_size)

    train_dl, val_dl = get_train_val_dataloader(args.source_dir, args.batch_size, max_shape, .1)

    vgae = get_vgae(max_shape, args.latent_size, device,
                    args.enc_size, args.dec_size,
                    args.num_gnn, args.num_expansions, args.dropout)
    optim = torch.optim.Adam(vgae.parameters(), lr=args.lr, weight_decay=args.wd)
    mae_loss = torch.nn.L1Loss()

    train(vgae, train_dl, val_dl, args.epoch_num, args.batch_size,
          optim, mae_loss, args.pres, args.beta, args.dest,
          device)

if __name__ == "__main__":
    main()
