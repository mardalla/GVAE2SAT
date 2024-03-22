import os
from subprocess import call
from argparse import ArgumentParser

def main():
    parser = ArgumentParser()
    parser.add_argument("cnf_dir", type=str,
                        help="path to directory of CNFs")
    args = parser.parse_args()

    for cnf in os.listdir(args.cnf_dir):
        path = os.path.join(args.cnf_dir, cnf)
        with open(path, "r") as file:
            lines = file.readlines()
        comments = []
        while lines[0][0] == 'c':
            comments.append(lines[0])
            lines = lines[1:]
        header = lines.pop(0).split()
        lines = [clause for clause in lines if len(clause) > 2]
        if len(lines) != int(header[3]):
            call(["rm", path])
            rename = cnf.split('_')
            rename[3] = f"{len(lines)}"
            rename = '_'.join(rename)
            print(f"{cnf} changed to {rename}.")
            path = os.path.join(args.cnf_dir, rename)
            header = " ".join(header[:3] + [f"{len(lines)}\n"])
            with open(path, "w") as file:
                file.writelines(comments)
                file.writelines([header])
                file.writelines(lines)
                file.writelines(["\n"])


if __name__ == "__main__":
    main()
