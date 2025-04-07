from src.model import UnetAttn
from diffusion.ddpm import Diffusion
from src.dataset import build_dataloader
from src.visualize import save_images, visualize_loss
import os
import torch
import torch.nn as nn
from tqdm import tqdm
from config_args import setup_parse, update_config, setup_logger, setup_logging
from datetime import datetime

def main(args):
    
    logger, log_file = setup_logger(args.log_dir)
    setup_logging(args.save_dir)

    logger.info(f"Training started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    logger.info(f"Batch size: {args.batch}, Image size: {args.imgsz}x{args.imgsz}\n")
    logger.info(f"Learning rate: {args.lr}, Image size: {args.imgsz}x{args.imgsz}\n")
    logger.info(f"Total number of epochs: {args.epochs}\n")

    device = args.devices

    model = UnetAttn(in_chans=args.in_chans,
                     out_chans=args.out_chans,
                     hidden_dim=args.hidden_dim,
                     time_dim=args.time_dim,
                     n_heads=args.n_heads,
                     is_attn=args.is_attn,
                     device=args.device)
    
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
    
    criterion = nn.MSELoss()

    diffusion = Diffusion(noise_steps=args.noise_steps,
                          beta_start=args.beta_start,
                          beta_end=args.beta_end,
                          imgsz=args.imgsz,
                          device=device)

    trainloader, testloader = build_dataloader(args)

    if args.debug:
        torch.autograd.set_detect_anomaly(True)

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(args.epochs):
        model.train()
        total_train_loss = 0

        pbar = tqdm(trainloader, desc=f"Epoch {epoch+1}/{args.epochs}", leave=False)

        for i, (images, _) in enumerate(pbar):
            images = images.to(device)
            t = diffusion.sample_timesteps(images.shape[0]).to(device)
            x_t, noise = diffusion.noise_images(images, t)
            
            predicted_noise = model(x_t, t)
            loss = criterion(noise, predicted_noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            pbar.set_postfix(MSE=loss.item())
        
        avg_train_loss = total_train_loss / len(trainloader)
        logger.info(f"[Epoch {epoch+1}] Train Loss: {avg_train_loss:.4f}")

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, _ in testloader:
                images = images.to(device)
                t = diffusion.sample_timesteps(images.shape[0]).to(device)
                x_t, noise = diffusion.noise_images(images, t)
                predicted_noise = model(x_t, t)
                loss = criterion(noise, predicted_noise)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(testloader)
        logger.info(f"[Epoch {epoch+1}] Validation Loss: {avg_val_loss:.4f}")

        
        sampled_images = diffusion.sample(model, n=images.shape[0])
        save_images(sampled_images, os.path.join(args.save_dir, "results", f"{epoch}.jpg"))

        # Scheduler step
        if args.scheduler == "plateau":
            scheduler.step(avg_val_loss)
        elif scheduler:
            scheduler.step()

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(args.save_dir, "models", f"ckpt_best.pt"))
            logger.info(f"Best model saved at epoch {epoch+1}.")
        else:
            patience_counter += 1
            logger.info(f"EarlyStopping counter: {patience_counter}/{args.early_stop_patience}")
            if patience_counter >= args.early_stop_patience:
                logger.info("Early stopping triggered.")
                break
    
    logger.info(f"\nTraining finished at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")


if __name__ == "__main__":
    
    parser = setup_parse()
    
    args = parser.parse_args()
    args = update_config(args)

    main(args)
