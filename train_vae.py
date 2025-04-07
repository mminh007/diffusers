import torch
from src.model import ResNetVAEV2
from src.dataset import build_dataloader
from src.loss import VAE_loss
from src.visualize import visualize_loss
import argparse
from tqdm import tqdm
import os
from datetime import datetime
import numpy as np
import logging


def setup_parse():
    parser = argparse.ArgumentParser()

    parser.add_argument("--in-chans", default=3, type=int)
    parser.add_argument("--num-chans", default=64, type=int)
    parser.add_argument("--out-chans", default=3, type=int)
    parser.add_argument("--z-dim", default=4, type=int)
    parser.add_argument("--embed-dim", default=4, type=int)
    parser.add_argument("--blocks", default=2, type=int)
    parser.add_argument("--channel-multipliers", nargs="+", type=int, default=[1,2,4,4])

    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--imgsz", default=32, type=int)

    parser.add_argument("--scheduler", type=str, default="step", choices=["step", "cosine", "plateau"])
    parser.add_argument("--step-size", type=int, default=5, help="Step size for StepLR")
    parser.add_argument("--gamma", type=float, default=0.5, help="Decay factor for StepLR and ReduceLROnPlateau")

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
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()
    return args


def setup_logger(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_file = os.path.join(log_dir, f"train_log_{timestamp}.log")

    logger = logging.getLogger("VAE_Training")
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)

    if logger.hasHandlers():
        logger.handlers.clear()
    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger, log_file


def main():

    args = setup_parse()

    logger, log_file = setup_logger(args.log_dir)

    logger.info("****** Training started ******")
    logger.info(f" Batch size: {args.batch}")
    logger.info(f" Image size: {args.imgsz}x{args.imgsz}")
    logger.info(f" Total number of epochs: {args.epochs}")

    trainloader, testloader = build_dataloader(args)

    model = ResNetVAEV2(in_chans=args.in_chans,
                        num_chans=args.num_chans,
                        out_chans=args.out_chans,
                        z_dim=args.z_dim,
                        embed_dim=args.embed_dim,
                        blocks=args.blocks,
                        channel_multipliers=args.channel_multipliers)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)

    if args.optimizer.lower() == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                    betas=[args.beta1, args.beta2], weight_decay=args.weight_decay)
    elif args.optimizer.lower() == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                    betas=[args.beta1, args.beta2], weight_decay=args.weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {args.optimizer}. Choose from ['adam', 'adamw'].")

        
    if args.scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    elif args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    elif args.scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=args.gamma, patience=3)
    else:
        scheduler = None

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

    if args.debug:
        torch.autograd.set_detect_anomaly(True)
        
    best_loss = np.inf
    patience_counter = 0

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

        logger.info(f"Epoch {epoch+1}:\n"
                    f"  Train_loss: {avg_train_loss:.6f}, Train_Reconstruction_loss: {recons_loss.item():.6f}, Train_KL_loss: {kl_loss.item():.6f}\n"
                    f"  Val_loss: {avg_val:.6f}, Val_Reconstruction_loss: {avg_recons:.6f}, Val_KL_loss: {avg_kl:.6f}\n")
        
        if scheduler is not None:
            if args.scheduler == "plateau":
                scheduler.step(avg_val)
            else:
                scheduler.step()

        os.makedirs(args.save_dir, exist_ok=True)
        ckpt_dir = os.path.join(args.save_dir, f"checkpoint_cifar10_{args.imgsz}.pth")
        
        if avg_val < best_loss:
            best_loss = avg_val
            torch.save(model.state_dict(), ckpt_dir)

            logger.info(f"****** Best model saved at epoch {epoch+1} ******")        
            patience_counter = 0
        else:
            patience_counter += 1
            logger.info(f"Early stopping patience: {patience_counter}/{args.early_stop_patience}")
            print(f"Early stopping patience: {patience_counter}/{args.early_stop_patience}")

        if patience_counter >= args.early_stop_patience:
            with open(log_file, "a") as f:
                logger.info(f"\n{'*' * 90}\n"
                        f"{' ' * 15} Early stopping triggered. Training stopped at epoch {epoch - args.early_stop_patience}.\n")
            print("Early stopping triggered. Training stopped.")
            break
    
    logger.info(f"\nTraining finished at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

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




    
