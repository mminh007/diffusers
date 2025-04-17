import torch
import numpy as np
import random
from torchvision import transforms
from datasets import load_dataset


def build_dataset(tokenizer, accelerator, args):
    
    if args.dataset_name is not None:
        dataset = load_dataset(args.dataset_name, args.dataset_config_name,
                               cache_dir=args.cache_dir, split="train")
        
    else:
        data_files = {}
        if args.train_data_dir is not None:
            data_files = args.train_data_dir
        dataset = load_dataset(
            "imagefolder",
            data_files=data_files,
            cache_dir=args.cache_dir,
        )

    column_names = dataset.column_names

    if args.image_column is None:
        image_column = column_names[0]
    else:
        image_column = args.image_column

        if image_column not in column_names:
            raise ValueError(
                f"Column '{image_column}' not found in dataset. Available columns: {', '.join(column_names)}"
            )
        
    if args.caption_column is None:
        caption_column = column_names[1]
    else:
        caption_column = args.caption_column

        if caption_column not in column_names:
            raise ValueError(
                f"Column '{caption_column}' not found in dataset. Available columns: {', '.join(column_names)}"
            )
    
    train_transforms = transforms.Compose([
        transforms.Resize((args.resolution, args.resolution), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.RandomHorizontalFlip() if args.random_flip else transforms.Lambda(lambda x: x),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    def tokenize_captions(examples, tokenizer, is_train=True):
        captions = []
        for caption in examples[caption_column]:
            if isinstance(caption, str):
                captions.append(caption)
            elif isinstance(caption, (list, np.ndarray)):
                # take a random caption if there are multiple
                captions.append(random.choice(caption) if is_train else caption[0])
            else:
                raise ValueError(
                    f"Caption column `{caption_column}` should contain either strings or lists of strings."
                )
        inputs = tokenizer(
            captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        return inputs.input_ids

    def preprocess_train(examples):
        images = [image.convert("RGB") for image in examples[image_column]]
        examples["pixel_values"] = [train_transforms(image) for image in images]
        examples["input_ids"] = tokenize_captions(examples, tokenizer)
        return examples

    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        input_ids = torch.stack([example["input_ids"] for example in examples])
        return {"pixel_values": pixel_values, "input_ids": input_ids}

    if accelerator.is_local_main_process:
        if args.max_train_samples is not None:
            dataset = dataset.shuffle(seed=args.seed).select(range(args.max_train_samples))
        train_dataset = dataset.with_transform(preprocess_train)
        #train_dataset.set_format(type="torch", columns=["pixel_values", "input_ids"])


    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    return train_dataloader