import torch

def remove_repeated_disjunctions_of_one(dictionary: dict) -> dict:
    """
      Allows no more than one unit clause consisting of a given variable.

    Parameter:
      <dictionary> := mapping of keys to individual clauses represented as
                      lists

    Returns:
      <dictionary> := the input dictionary, but, when considering only
                      the key-value pairs with singleton values,  there is now
                      an injection - i.e., no two keys map to lists of only 
                      one element where the element is the same
    """
    # Find key-value pairs with singleton values.
    repeats = [dictionary[x][0] for x in dictionary.keys() if len(dictionary[x]) == 1]
    # Count singletons with more than one occurrence.
    counts = [(x, repeats.count(x)) for x in set(repeats) if repeats.count(x) > 1]

    # Delete all but one of the keys mapping to repeated singletons.
    for repeat in counts:
        count = repeat[1]
        delenda = []
        for key in dictionary.keys():
            if dictionary[key] == [repeat[0]]:
                delenda.append(key)
                count -= 1
            if count == 1:
                break

        for d in delenda:
            del dictionary[d]

    # Return the trimmed version.
    return dictionary


def adj_to_cnf(matrix: torch.Tensor,
               desig,
               file_to_write: str,
               polarity_ratio: float | None = None,
               indices: torch.Tensor | None = None,
               comments: list[str] | None = None) -> dict:
    """
      Converts an adjacency matrix into CNF. As a side effect, saves a file
      in DIMACS. Returns a dictionary of clauses.

    Parameters: 
      <matrix>         := adjacency matrix to be converted
      <desig>          := positive and negative thresholds for inclusion
                          or
                          scalar number to include
      <file_to_write>  := where to save the DIMACS output
      <polarity_ratio> := #(positive literals) / #(negative literals)
                              considered only if <desig> is scalar
      <indices>        := optional, precalculated index matrix - sparing
                              calculation on each call
      <comments>       := optional comments to start DIMACS file
    """
    # Create flat index for 2D tensor.
    if indices is None:
        indices = torch.arange(matrix.size(0)*matrix.size(1)).reshape(matrix.size(0), -1)
    
    clauses = {}
    variables = set()

    try:
        # Positive literals
        for x in indices[matrix >= desig[0]]:
            x = x.item()
            clause = x // matrix.size(1)
            var = (x % matrix.size(1)) + 1
            variables = variables.union([var])
            if clause in clauses.keys():
                clauses[clause].append(var)
            else:
                clauses[clause] = [var]

        # Negative literals
        for x in indices[matrix <= desig[1]]:
            x = x.item()
            clause = x // matrix.size(1)
            var = (x % matrix.size(1)) + 1
            variables = variables.union([var])
            if clause in clauses.keys():
                clauses[clause].append(-var)
            else:
                clauses[clause] = [-var]

    except:
        # forcing
        if polarity_ratio is not None:
            desig = int(desig)
            if polarity_ratio == 0:
                desig = [0, desig]
            elif polarity_ratio == -1:
                desig = [desig, 0]
            else:
                num_negative_literals = round(desig / (polarity_ratio+1))
                desig = [desig - num_negative_literals, num_negative_literals]

            flat = torch.flatten(matrix)
            for x in flat.sort(descending=True).indices[:desig[0]]:
                x = x.item()
                clause = x // matrix.size(1)
                var = (x % matrix.size(1)) + 1
                variables = variables.union([var])
                if clause in clauses.keys():
                    clauses[clause].append(var)
                else:
                    clauses[clause] = [var]
            for x in flat.sort(descending=False).indices[:desig[1]]:
                x = x.item()
                clause = x // matrix.size(1)
                var = (x % matrix.size(1)) + 1
                variables = variables.union([var])
                if clause in clauses.keys():
                    clauses[clause].append(-var)
                else:
                    clauses[clause] = [-var]
        
        # scalar desig
        else:
            flat = torch.flatten(matrix)
            for x in abs(flat).sort(descending=True).indices[:int(desig)]:
                x = x.item()
                clause = x // matrix.size(1)
                var = (x % matrix.size(1)) + 1
                variables = variables.union([var])
                if flat[x] < 0:
                    var = -var
                if clause in clauses.keys():
                    clauses[clause].append(var)
                else:
                    clauses[clause] = [var]

    clauses = remove_repeated_disjunctions_of_one(clauses)
    compressed_variables = {}
    for idx, var in enumerate(variables):
        compressed_variables[var] = idx + 1
    
    with open(file_to_write, "w") as file:
        if comments is not None:
            file.writelines(comments)
        # header
        file.write(f"p cnf {len(variables)} {len(clauses)}\n")
        for clause in clauses:
            file.write(" ".join([f"{compress_variable(var, compressed_variables)}"
                                 for var in clauses[clause]]) + " 0\n")

    return clauses


def mask_variables(matrix, num_kept_vars):
    device = matrix.device
    maxima = abs(matrix).max(dim=0).values
    kept_cols = maxima.sort(descending=True).indices[:num_kept_vars]
    return torch.hstack([matrix[:, kept_cols],
                         torch.zeros([matrix.size(0), matrix.size(1)-num_kept_vars], device=device)])


def mask_clauses(matrix, num_kept_rows):
    device = matrix.device
    maxima = abs(matrix).max(dim=1).values
    kept_rows = maxima.sort(descending=True).indices[:num_kept_rows]
    return torch.vstack([matrix[kept_rows],
                         torch.zeros([matrix.size(0)-num_kept_rows, matrix.size(1)], device=device)])


def compress_variable(var, compressed_variables):
    var = compressed_variables[var] if var > 0 else -(compressed_variables[-var])
    return var
