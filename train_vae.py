import torch
from VAE.model import ResNetVAE 
from VAE.dataset import build_dataloader
from VAE.loss import VAE_loss
from VAE.visualize import visualize_loss
import argparse
from tqdm import tqdm
import os
from datetime import datetime
import numpy as np


def setup_parse():
    parser = argparse.ArgumentParser()

    parser.add_argument("--in-chans", default=3, type=int)
    parser.add_argument("--latent-dim", default=128, type=int)

    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--imgsz", default=32, type=int)

    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--optimizer", default="adamw", type=str)
    parser.add_argument("--beta1", default=0.9, type=float)
    parser.add_argument("--beta2", default=0.999, type=float)
    parser.add_argument("--weight-decay", default=0.05, type=float)
    parser.add_argument("--batch", default=16, type=int)
    parser.add_argument("--num-workers", default=1, type=int)

    # config loss
    parser.add_argument("--B", type=int, default=100)
    parser.add_argument("--reduction", type=str, default="sum")

    parser.add_argument("--log-dir", type=str, default="logs")
    parser.add_argument("--save-dir", type=str, default="checkpoints")
    parser.add_argument("--early-stop-patience", type=int, default=5)
    parser.add_argument("--visualize", action="store_true")


    args = parser.parse_args()
    return args


def main():

    args = setup_parse()

    os.makedirs(args.log_dir, exist_ok=True)

    timestap = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_file = os.path.join(args.log_dir, f"train_log_{timestap}.txt")
    

    with open(log_file, "w") as f:
        f.write(f"Training started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model: VAE with latent_dim={args.latent_dim}\n")
        f.write(f"Batch size: {args.batch}, Image size: {args.imgsz}x{args.imgsz}\n")
        f.write(f"Total number of epochs: {args.epochs}\n")

    trainloader, testloader = build_dataloader(args)

    model = ResNetVAE(in_chans=args.in_chans,
                      out_chans=args.in_chans,
                      latent_dim=args.latent_dim)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)

    if args.optimizer is not None:
        if args.optimizer.lower() == "adam":
            optimizer = torch.optim.Adam(model.parameters(), lr = args.lr,
                                         betas=[args.beta1, args.beta2], weight_decay=args.weight_decay)
        
        elif args.optimizer.lower() == "adamw":
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                          betas=[args.beta1, args.beta2], weight_decay=args.weight_decay)        
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                          betas=[args.beta1, args.beta2], weight_decay=args.weight_decay)
        
    
    # if args.dtype.lower() == "bf16":
    #     dtype = torch.bfloat16
    # elif args.dtype.lower() == "fp16":
    #     dtype = torch.float16
    # else:
    #     dtype = torch.float32

    criterion = VAE_loss(reduction=args.reduction, B=args.B)

    "--------------------------------------------------------------------------------------------"
    "-------------------------------- TRAINING TIME----------------------------------------------"
    "--------------------------------------------------------------------------------------------"
    
    if args.visualize:
        train_loss_history = []
        train_recon_loss_history = []
        train_kl_loss_history = []
        val_loss_history = []
        val_recon_loss_history = []
        val_kl_loss_history = []

    torch.autograd.set_detect_anomaly(True)
    best_loss = np.inf

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        running_recon_loss = 0.0
        running_kl_loss = 0.0

        epoch_bar = tqdm(trainloader, desc=f"Epoch {epoch+1}/{args.epochs}", leave=False)
        for x, _ in epoch_bar:
            x = x.to(device)
            #y = y.to(device).long()

            optimizer.zero_grad()
            z, mu, logvar = model(x)

            loss, recons_loss, kl_loss = criterion(z, x, mu, logvar)
            loss.backward()
            running_loss += loss.item()
            running_recon_loss += recons_loss.item()
            running_kl_loss += kl_loss.item()

            optimizer.step()

        avg_train_loss = running_loss / len(trainloader)
        avg_train_recons_loss = running_recon_loss / len(trainloader)
        avg_train_kl_loss = running_kl_loss / len(trainloader)
        epoch_bar.set_postfix(loss=avg_train_loss)

        if args.visualize:
            train_loss_history.append(avg_train_loss)
            train_recon_loss_history.append(avg_train_recons_loss)
            train_kl_loss_history.append(avg_train_kl_loss)

        print(f"Epoch {epoch+1}/{args.epochs}: Train_Loss: {avg_train_loss} "
                f"Reconstruction Loss: {avg_train_recons_loss}, KL Divergence: {avg_train_kl_loss}\n")
        
        with open(log_file, "a") as f:
            f.write(f"Epoch {epoch+1}:\n"
                    f"  Train_loss: {avg_train_loss:.6f}, Train_Reconstruction_loss: {recons_loss.item():.6f}, Train_KL_loss: {kl_loss.item():.6f}\n")

        model.eval()
        val_loss = 0.0
        val_recons = 0.0
        val_kl = 0.0
        
        with torch.no_grad():
            for images, _ in testloader:
                images = images.to(device)
                z, mu, logvar = model(images)
                loss, recons_loss, kl_loss = criterion(z, x, mu, logvar)

                val_loss += loss.item()
                val_recons += recons_loss.item()
                val_kl += kl_loss.item()

        avg_val = val_loss / len(testloader)
        avg_recons = val_recons / len(testloader)
        avg_kl = val_kl / len(testloader)

        if args.visualize:
            val_loss_history.append(avg_val)
            val_recon_loss_history.append(avg_recons)
            val_kl_loss_history.append(avg_kl)

        print(f"=== Val_loss: {avg_val:.6f}, Val Reconstruction Loss: {avg_recons:.6f}, Val KL Divergence: {avg_kl:.6f}\n")

        with open(log_file, "a") as f:
            f.write(f"  Val_loss: {avg_val:.6f}, Val_Reconstruction_loss: {avg_recons:.6f}, Val_KL_loss: {avg_kl:.6f}\n")

        os.makedirs(args.save_dir, exist_ok=True)
        ckpt_dir = os.path.join(args.save_dir, f"checkpoint_cifar10_{args.imgsz}.pth")
        
        if avg_val < best_loss:
            best_loss = avg_val
            torch.save(model.state_dict(), ckpt_dir)           
            print(f"Best model saved at epoch {epoch+1}")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"Early stopping patience: {patience_counter}/{args.early_stop_patience}")

        if patience_counter >= args.early_stop_patience:
            with open(log_file, "a") as f:
                f.write(f"\n{'*' * 90}\n"
                        f"{' ' * 15} Early stopping triggered. Training stopped at epoch {epoch - args.early_stop_patience}.\n")
            print("Early stopping triggered. Training stopped.")
            break
    
    with open(log_file, "a") as f:
        f.write(f"\nTraining finished at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    if args.visualize:
        losses_dict = {
            "train_loss_history": train_loss_history,
            "train_recon_loss_history": train_recon_loss_history,
            "train_kl_loss_history": train_kl_loss_history,
            "val_loss_history": val_loss_history,
            "val_recon_loss_history": val_recon_loss_history,
            "val_kl_loss_history": val_kl_loss_history 
        }

        visualize_loss(losses_dict)

if __name__ == "__main__":
    main()




    
