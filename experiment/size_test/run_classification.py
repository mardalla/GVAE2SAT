from argparse import ArgumentParser
import sys

import pandas as pd
import torch

sys.path.append("../..")

from classify import classify, train_test_labels_split, train_then_classify
from GWAE2SAT.data import get_max_shape


def main():
    parser = ArgumentParser()
    parser.add_argument("--src", dest="src", type=str,
                        help="path to CNFs", default="/home/dcrowley/cnfs/cnfs_train")
    parser.add_argument("--dest", dest="dest", type=str,
                        default="classification_results.csv",
                        help="where to save results", )
    args = parser.parse_args()

    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    
    max_shape = get_max_shape(args.src)
    
    (X_train_names, y_train_names,
     X_test_names, y_test_names) = train_test_labels_split(args.src, .1)

    dictionary = {}
    dictionary["full"] = [classify(X_train_names, y_train_names,
                                   X_test_names, y_test_names,
                                   max_shape, args.src, device)]
    print(f"Full: {dictionary['full'][0]}")
    dictionary["32"] = [train_then_classify(X_train_names, y_train_names,
                                            X_test_names, y_test_names,
                                            32, max_shape, device,
                                            128, 256, args.src)]
    print(f"  32: {dictionary['32'][0]}")
    dictionary["1024"] = [train_then_classify(X_train_names, y_train_names,
                                              X_test_names, y_test_names,
                                              1024, max_shape, device,
                                              1024, 1024, args.src)]
    print(f"1024: {dictionary['1024'][0]}")
    pd.DataFrame(dictionary).to_csv(args.dest, index=False)


if __name__ == "__main__":
    main()
