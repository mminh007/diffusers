import yaml
import argparse
import os
import logging
from datetime import datetime


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")
    

def setup_parse():
    parser = argparse.ArgumentParser()

    # Unet config
    parser.add_argument("--config-file", type=str, help="path to config file")
    parser.add_argument("--in-chans", type=int, help="dimension of input")
    parser.add_argument("--imgsz", type=int, help="size of input")
    parser.add_argument("--out-chans", default=3, type=int)
    parser.add_argument("--hidden-dim", default=64, type=int)
    parser.add_argument("--time-dim", default=256, type=int)
    parser.add_argument("--act", default="swish", type=str)
    parser.add_argument("--dropout", default=0.3, type=float)
    parser.add_argument("--n-heads", default=1, type=int)
    parser.add_argument("--qkv-bias", action="store_true")
    parser.add_argument("--qk-scale", default=None, type=float)
    parser.add_argument("--is-attn", nargs="+", type=str2bool, default=[False, False, True, True])
    parser.add_argument("--down-scale", default=3, type=int)
    parser.add_argument("--residual", action="store_true")
    
    # Diffusion config
    parser.add_argument("--noise-steps", default=1000, type=int)
    parser.add_argument("--beta-start", default=1e-4, type=float)
    parser.add_argument("--beta-end", default=0.02, type=float)

    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--batch", default=16, type=int)
    parser.add_argument("--devices", default="cuda", type=str)
    parser.add_argument("--optimizer", default="adamw", type=str)
    parser.add_argument("--weight-decay", default=0.05, type=float)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--beta1", default=0.9, type=float)
    parser.add_argument("--beta2", default=0.999, type=float)
    parser.add_argument("--num-workers", default=1, type=int)

    parser.add_argument("--scheduler", type=str, default="step", choices=["step", "cosine", "plateau"])
    parser.add_argument("--step-size", type=int, default=5, help="Step size for StepLR")
    parser.add_argument("--gamma", type=float, default=0.5, help="Decay factor for StepLR and ReduceLROnPlateau")

    parser.add_argument("--log-dir", type=str, default="logs")
    parser.add_argument("--save-dir", type=str, default="checkpoints")
    parser.add_argument("--early-stop-patience", type=int, default=5)
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--debug", action="store_true")

    return parser


def update_config(args: argparse.Namespace):
    if not args.config_file:
        return args
    
    cfg_path = args.config_file + ".yaml" if not args.config_file.endswith(".yaml") else args.config_file

    with open(cfg_path, "r") as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    
    for key, value in data.items():
        if getattr(args, key) is None:
            setattr(args, key, value)

    # config_args = argparse.Namespace(**data)
    # args = parser.parse_args(namespace=config_args)

    return args


def setup_logging(run_name):
    os.makedirs(run_name, exist_ok=True)
    os.makedirs(os.path.join(run_name, "models"), exist_ok=True)
    os.makedirs(os.path.join(run_name, "results"), exist_ok=True)


def setup_logger(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_file = os.path.join(log_dir, f"train_log_{timestamp}.log")

    logger = logging.getLogger("Diffusion_Training")
    logger.setLevel(logging.INFO)

    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)

    # Stream handler (console)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_formatter = logging.Formatter('%(message)s')
    stream_handler.setFormatter(stream_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger, log_file


