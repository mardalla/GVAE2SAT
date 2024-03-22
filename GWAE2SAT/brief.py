import os
from pysat.formula import CNF
from pysat.solvers import Glucose42

def main():
    problems = [p for p in os.listdir("generations") if p.endswith("0.cnf")]
    problems = [p for p in problems if p.startswith("sat")]

    sat = 0
    for prob in problems:
        cnf = CNF(from_file=os.path.join("generations", prob))
        solver = Glucose42(bootstrap_with=cnf.clauses)
        solved = solver.solve()
        sat += solved
        print(solved)
    print(sat)

if __name__ == "__main__":
    main()
