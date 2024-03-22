from argparse import ArgumentParser

import torch

from generate import generate


def main():
    parser = ArgumentParser()
    parser.add_argument("--src", dest="source_dir", type=str,
                        default="../cnfs/cnfs_test",
                        help="path of directory with CNFs to regenerate")
    parser.add_argument("--dest", dest="dest", type=str, default="generations",
                        help="directory to save new CNFs in")
    parser.add_argument("--state_dict", dest="state_dict", type=str,
                        help="location of trained model's weights",
                        default="models/max_shape[2820, 250]_ep1_bat6_lr0.003_z32_pres9_beta1.0.pt")
    # features to specify trained model without name
    parser.add_argument("--max_shape", dest="max_shape", type=list[int],
                        default=[2820, 250])
    parser.add_argument("--epoch_num", dest="ep", type=int, default=16)
    parser.add_argument("--batch_size", dest="bat", type=int, default=16)
    parser.add_argument("--lr", dest="lr", type=float, default=.001)
    parser.add_argument("--wd", dest="wd", type=float, default=0)
    parser.add_argument("--z", dest="z", type=int, default=32)
    parser.add_argument("--pres", dest="pres", type=int, default=9)
    parser.add_argument("--beta", dest="beta", type=float, default=1)

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
    parser.add_argument("--regen_factor", dest="regen_factor", type=float, default=1,
                        help="how big the new instance should be relative to the original")
    parser.add_argument("--mask_by", dest="mask", choices=["clause", "var", None], default="clause",
                        help='if "clause" or "var", reproduce the original\'s clause or variable number')
    parser.add_argument("--num_gen", dest="num_gen", type=int, default=10,
                        help="number of generations to make per instance")
    parser.add_argument("--pol_rat", dest="pol_rat", type=bool, default=True,
                        help="whether to force the original proportion of true to false literals")
    args = parser.parse_args()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    torch.manual_seed(1)

    if args.state_dict is None:
        state_dict = (f"max_shape{args.max_shape}_ep{args.ep}_bat{args.bat}_lr{args.lr}_"
                      + (f"wd{args.wd}_" if args.wd > 0 else "")
                      + f"z{args.z}_pres{args.pres}_beta{args.beta}.pt")
    else:
        state_dict = args.state_dict

    generate(state_dict, args.source_dir, args.dest, args.bat, device,
             args.mask, args.regen_factor, args.enc_size, args.dec_size,
             args.num_gnn, args.num_expansions, args.dropout, args.num_gen,
             polarity_ratio=args.pol_rat)
    

if __name__ == "__main__":
    main()
