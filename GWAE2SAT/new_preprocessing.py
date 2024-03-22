import torch

def cnf_to_adj(file_to_read: str,
               reduced=True) -> torch.Tensor:
    """
      Converts from DIMACS to Torch tensor.

    Parameter:
      <file_to_read> := path to file with CNF in DIMACS format
      <reduced>      := whether to return only one corner of the adjacency
                        matrix
     
    Returns:
      <adj>          := Torch tensor representing the Clause x Variable
                        corner of the adjacency matrix of a weighted
                        Variable-Clause Graph of the problem in 
                        <file_to_read>
                        Weights are +1 for positive literals
                                    -1 for negative literals
    """
    with open(file_to_read, "r") as file:
        cnf = file.readlines()
    # Ignore comment lines (beginning with 'c').
    cnf = list(filter(lambda x: x[0] != 'c', cnf))
    # Separate header (beginning with 'p') from body.
    header = cnf.pop(0).strip("\n ").split()
    # Read the variable and clause numbers from the header.
    num_var = int(header[2])
    num_clauses = int(header[3])

    # Clause x Var section of whole VCG adjacency matrix
    adj = torch.zeros(num_clauses+num_var, num_clauses+num_var)

    # Populate Weighted VCG.
    # Weights: +1 for positive literal
    #          -1 for negative literal
    for id, clause in enumerate(cnf):
        for lit in clause.split()[:-1]:
            lit = int(lit)
            if lit < 0:
                adj[id, -lit - 1 + num_clauses] = -1
            else:
                adj[id, lit - 1 + num_clauses] = 1

    # Fill symmetric adjacency matrix.
    if not reduced:
        full = torch.zeros(num_clauses+num_var, num_clause+num_var)
        full[:num_clauses, -num_vars:] = adj
        full[-num_vars:, :num_clauses] = adj.T
        adj = full

    adj += torch.eye(num_clauses+num_var)
    
    return adj


def pad_smaller_instance(instance: torch.Tensor,
                         goal_size: list[int]) -> torch.Tensor:
    """
      Pads smaller tensor with 0s until it has a required size.

    Parameters:
      <instance>  := Torch matrix to be enlargened
      <goal_size> := list specifying how many rows and columns
                     the output should have

    Returns:
      <instance>  := Torch matrix with size = torch.Size(<goal_size>)
    """
    # Matrix of 0s required to make input matrix as wide as
    # desired
    required_cols = goal_size[1] - instance.size(1)
    pad_right = torch.zeros(instance.size(0), required_cols)

    # Matrix of 0s required to make widened matrix as
    # tall as desired
    required_rows = goal_size[0] - instance.size(0)
    pad_below = torch.zeros(required_rows, goal_size[1])

    instance = torch.cat((instance, pad_right), dim=1)
    instance = torch.cat((instance, pad_below), dim=0)

    return instance
