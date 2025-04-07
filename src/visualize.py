import matplotlib.pyplot as plt
from IPython.display import display
from PIL import Image
import torch
import torchvision


def visualize_loss(losses: dict):
    """
    losses_dict = {
            "train_loss_history": train_loss_history,
            "train_recon_loss_history": train_recon_loss_history,
            "train_kl_loss_history": train_kl_loss_history,
            "val_loss_history": val_loss_history,
            "val_recon_loss_history": val_recon_loss_history,
            "val_kl_loss_history": val_kl_loss_history 
        }
    """
    epochs = range(1, len(losses["train_loss_history"]) + 1)
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(epochs, losses["train_loss_history"], label='Train Loss')
    plt.plot(epochs, losses["val_loss_history"], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Total Loss')
    plt.legend()
    
    plt.subplot(1, 3, 2)
    plt.plot(epochs, losses["train_recon_loss_history"], label='Train Recon Loss')
    plt.plot(epochs, losses["val_recon_loss_history"], label='Validation Recon Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Reconstruction Loss')
    plt.legend()
    
    plt.subplot(1, 3, 3)
    plt.plot(epochs, losses["train_kl_loss_history"], label='Train KL Loss')
    plt.plot(epochs, losses["val_kl_loss_history"], label='Validation KL Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('KL Divergence Loss')
    plt.legend()
    
    plt.savefig("/loss_plot.jpg") 
    plt.show()
    plt.pause(5)

    display(Image.open("/loss_plot.jpg"))


def plot_images(images):
    plt.figure(figsize = (32,32))
    plt.imshow(torch.cat([
        torch.cat([i for i in images.cpu()], dim=-1)
        ], dim=-2).permute(1, 2, 0).cpu())
    plt.show()



def save_images(images, path, **kwargs):
    grid = torchvision.utils.make_grid(images, **kwargs)
    ndarr = grid.permute(1, 2, 0).to("cpu").numpy()
    im = Image.fromarray(ndarr)
    im.save(path)