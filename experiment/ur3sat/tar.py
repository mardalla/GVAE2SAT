from argparse import ArgumentParser
import os
import subprocess

def main():
    parser = ArgumentParser()
    parser.add_argument("--dest", dest="dest")
    args = parser.parse_args()
    
    if not os.path.isdir(args.dest):
        os.mkdir(args.dest)
    
    for x in os.listdir():
        if x.endswith("tar.gz"):
            subprocess.call(["tar", "-xzf", x,
                             "-C", args.dest])
    os.chdir(args.dest)
    
    for d in os.listdir():
        if d.startswith("UU"):
            for f in os.listdir(d):
                subprocess.call(["mv", f"{d}/{f}", '.'])
            subprocess.call(["rmdir", f"{d}"])
    for d in os.listdir("ai/hoos/Shortcuts"):
        here = f"ai/hoos/Shortcuts/{d}"
        for f in os.listdir(here):
            subprocess.call(["mv", f"{here}/{f}", '.'])
    for d in os.listdir("ai/hoos/Research/SAT/Formulae"):
        here = f"ai/hoos/Research/SAT/Formulae/{d}"
        for f in os.listdir(here):
            subprocess.call(["mv", f"{here}/{f}", '.'])

    subprocess.call(["rm", "-r", "ai"])

    for sat_instance in os.listdir():
        with open(sat_instance, "r") as file:
            lines = file.readlines()
        while lines[-1][0] != '%':
            lines = lines[:-1]
        lines = lines[:-1]
        with open(sat_instance, "w") as file:
            file.writelines(lines)
    

if __name__ == "__main__":
    main()
