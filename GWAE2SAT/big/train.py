from os.path import isdir, join
from subprocess import call
import torch

def train(vgae, train_dl, val_dl, epoch_num, bat,
          optim, mae_loss, pres, beta, dest,
          device):
    best_val_loss = float("inf")
    for epoch in range(epoch_num):
        epoch_loss = 0
        denominator = 0

        for i, data in enumerate(train_dl):
            data = data.to(device)
            batch_size = len(data)
            denominator += batch_size

            train_present_recon_loss = 0
            train_absent_recon_loss = 0
            train_kl_loss = 0

            for j, datum in enumerate(data):
                vgae.train()

                z = vgae.encode(datum)
                generation = vgae.decode(z)

                train_present_recon_loss += mae_loss(generation[datum != 0], datum[datum != 0])
                train_absent_recon_loss += mae_loss(generation[datum == 0], datum[datum == 0])
                train_kl_loss += vgae.kl_loss()

            train_recon_loss = ((pres*train_present_recon_loss + train_absent_recon_loss)
                                / (pres+1))
        
            train_loss = train_recon_loss + beta * train_kl_loss
            epoch_loss += train_loss
            print(f"{epoch}:{i}",
                  f"Training Reconstruction Loss: {train_recon_loss / batch_size}",
                  f"Training KL Loss: {train_kl_loss / batch_size}",
                  "",
                  sep="\n")

            train_loss.backward()
            optim.step()
            optim.zero_grad()

        epoch_train_loss = epoch_loss / denominator

        denominator = 0
        val_loss = 0
        with torch.no_grad():
            vgae.eval()

            for i, data in enumerate(val_dl):
                data = data.to(device)
                batch_size = len(data)
                denominator += batch_size

                val_present_recon_loss = 0
                val_absent_recon_loss = 0
                val_kl_loss = 0

                for j, datum in enumerate(data):
                    z = vgae.encode(datum)
                    generation = vgae.decode(z)
                
                    val_present_recon_loss += mae_loss(generation[datum != 0], datum[datum != 0])
                    val_absent_recon_loss += mae_loss(generation[datum == 0], datum[datum == 0])
                    val_kl_loss += vgae.kl_loss()

            
                val_recon_loss = ((pres*val_present_recon_loss + val_absent_recon_loss)
                                  / (pres+1))
                val_loss += val_recon_loss + beta * val_kl_loss
            
        val_loss = val_loss / denominator
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(vgae.state_dict(), "checkpoint.pt")
    
        print()
        print()
        print(f"Epoch: {epoch} Training Loss  : {epoch_train_loss}")
        print(f"       {len(str(epoch))*' '} Validation Loss: {val_loss}")
        print()
        print()

    if not isdir(dest):
        call(["mkdir", dest])
    wd = optim.param_groups[0]["weight_decay"]
    save_file = join(dest,
                     f"max_shape{list(generation.shape)}_"
                     + f"ep{epoch_num}_bat{bat}_lr{optim.param_groups[0]['lr']}_"
                     + (f"wd{wd}_" if wd > 0 else "")
                     + f"z{z.size(1)}_pres{pres}_beta{beta}.pt")
    call(["mv", "checkpoint.pt", save_file])
    print(f"{save_file} saved.")

    return save_file
