from os.path import isdir, join
from numpy.random import choice
from subprocess import call

import torch

from data import get_dataloader_with_names, parity_comparison
from model import get_vgae
from postprocessing import adj_to_cnf, mask_variables, mask_clauses


def generate(state_dict, src, dest, batch_size, device, mask, regen_factor,
             enc_size, dec_size, num_gnn, num_expansions, dropout, num_gen,
             labels=None, polarity_ratio=False):

    max_shape = state_dict[state_dict.find("max_") + 9:]
    max_shape = eval(max_shape[:max_shape.find(']')+1])
    latent_size = state_dict[state_dict.find('z')+1:]
    latent_size = int(latent_size[:latent_size.find('_')])
    vgae = get_vgae(max_shape, latent_size, device, enc_size, dec_size,
                    num_gnn, num_expansions, dropout)
    vgae.load_state_dict(torch.load(state_dict, map_location=device))

    original_cnfs, labels = get_dataloader_with_names(src, batch_size, max_shape, labels)

    if not isdir(dest):
        call(["mkdir", dest])

    indices = torch.arange(max_shape[0]*max_shape[1]).reshape(max_shape)
    latent_codes = []
    with torch.no_grad():
        # keep the code's \sigma in play
        # and make the model stochastic
        # vgae.eval()
        for idx, data in enumerate(original_cnfs):
            data = data.to(device)
            for jdx, datum in enumerate(data):
                original_name = labels[idx*batch_size + jdx]
                comments = ["c Generated from\n", f"c {original_name}\n", "c\n"]

                for generatum in range(num_gen):

                    z = vgae.encode(datum)
                    latent_codes += [z]
                    new_adj = vgae.decode(z)

                    if polarity_ratio:
                        (num_pos_lits,
                         num_neg_lits) = parity_comparison(join(src,
                                                                original_name))
                        if num_pos_lits == 0:
                            pol_ratio = 0
                        elif num_neg_lits == 0:
                            pol_ratio = -1
                        else:
                            pol_ratio = num_pos_lits / num_neg_lits
                    else:
                        pol_ratio = None
                    
                    if mask is not None:
                        with open(join(src, original_name), "r") as file:
                            header = "c"
                            while header[0] != 'p':
                                header = file.readline()
                        header = header.split()
                        num_clauses = int(header[3])
                        num_var = int(header[2])

                        # For larger matrix
                        new_adj = new_adj[:num_clauses,
                                          num_clauses:num_clauses+num_var]

                    original_num = abs(datum).sum()
                    original_num -= (num_clauses+num_var)
                    original_num /= 2
                    adj_to_cnf(new_adj, original_num*regen_factor,
                               join(dest, f"{original_name[:-4]}-{generatum}.cnf"),
                               pol_ratio, indices, comments)

                # # num_gen-1 more generations from slightly different latent codes
                # tweaks = choice(range(latent_size-1), num_gen-1, False)
                # for sample in range(1, num_gen):
                #     changed_latent = torch.clone(z)
                #     changed_latent[0, tweaks[sample-1]] += 1e-8
                #     new_adj = vgae.decode(z)
                        
                #     adj_to_cnf(new_adj, original_num*regen_factor,
                #                join(dest, f"{original_name[:-4]}-{sample}.cnf"),
                #                pol_ratio, indices, comments)
                    
                print(f"{original_name} done")
                    
    return latent_codes
