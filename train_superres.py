import argparse 
from time import time

import torch 
from torch.utils.data import DataLoader
from torch.nn.functional import avg_pool2d
from einops import reduce
from accelerate import Accelerator

from upsample import upsample_wavelet, wavelet_to_image
from image_et import (
    ImageET as ET,
    Patch,
    GetCIFAR,
    gen_mask_id,
    count_parameters,
    device,
    str2bool,
    get_latest_file,
)


def load_data(data_path, data_name): 
    """
        Method to load ImageNet1k Dataset 
    """
    ... 


def main(args): 
    IMAGE_FOLDER = args.result_dir + "/images"
    MODEL_FOLDER = args.result_dir + "/models"

    accelerator = Accelerator()
    device = accelerator.device

    # get args 
    batch_size = args.batch_size
    
    x = torch.randn(1, 3, 32, 32)
    patch_fn = Patch(dim=args.patch_size)
    model = ET(
        x,
        patch_fn,
        args.out_dim,
        args.tkn_dim,
        args.qk_dim,
        args.nheads,
        args.hn_mult,
        args.attn_beta,
        args.attn_bias,
        args.hn_bias,
        time_steps=args.time_steps,
        blocks=args.blocks,
        hn_fn=lambda x: -0.5 * (torch.relu(x) ** 2.0).sum(),
    )

    if accelerator.is_main_process:
        print(f"Number of parameters: {count_parameters(model)}", flush=True)

    NUM_PATCH = model.patch.N
    NUM_MASKS = int(model.patch.N * args.mask_ratio)

    # TODO: Change the dataset!
    trainset, testset, unnormalize_fn = GetCIFAR(args.data_path, args.data_name)

    train_loader, test_loader = map(
        lambda z: DataLoader(
            z,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            drop_last=True,
            pin_memory=True,
        ),
        (trainset, testset),
    )

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(args.b1, args.b2),
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        opt,
        T_0=10, 
        T_mult=1, 
        eta_min=1e-6
    )

    start_epoch = 1
    latest_checkpoint = get_latest_file(MODEL_FOLDER, ".pth")
    if latest_checkpoint is not None:
        if accelerator.is_main_process:
            print(f"Found latest checkpoint file: {latest_checkpoint}", flush=True)

        checkpoint = torch.load(latest_checkpoint, map_location="cpu")  
        start_epoch = checkpoint["epoch"]
        opt.load_state_dict(checkpoint["opt"])
        model.load_state_dict(checkpoint["model"])
        scheduler.load_state_dict(checkpoint["scheduler"])

    model, opt, train_loader, test_loader, scheduler = accelerator.prepare(
        model, opt, train_loader, test_loader, scheduler
    )

    visual_num = 16
    training_display = range(start_epoch, args.epochs + 1)

    for i in training_display:
        running_loss = 0.0

        model.train()

        start_time = time()
        for x, _ in train_loader:
            # downsample by factor 2 
            x_down = avg_pool2d(x, kernel_size=2)

            # upscale by factor 2 
            x_ups, mask_id = upsample_wavelet(x_down, scale_factor=2)
            batch_id = torch.arange(batch_size)[:, None]

            patched_x = patch_fn(x_ups)
            # Create mask for one patch
            mask = torch.ones(args.patch_size, args.patch_size, dtype=torch.bool)
            mask[::2, ::2] = False  # Every other pixel in alternate rows is unmasked

            # Flatten the mask to match patch dimension
            mask = mask.reshape(-1).repeat(3)  # Now it's [3*H*W]

            # Get the masked values
            y = patched_x[..., mask]  

            x, x_ups, y, batch_id, mask_id = map(
                lambda z: z.to(device), (x, x_ups, y, batch_id, mask_id)
            )

            pred = model(x_ups, mask=None, alpha=args.alpha)

            # grab recovered mask-tokens
            yh = pred[..., mask]

            loss = reduce((yh - y) ** 2, "b ... -> b", "mean").mean()
            accelerator.backward(loss)

            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), 1.0)

            opt.step()
            opt.zero_grad()
            running_loss += loss.item()
            print("one step done")

        torch.cuda.synchronize()
        end_time = time()
        scheduler.step(running_loss)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ET as Mask Auto-encoder")
    parser.add_argument("--global-seed", default=3407, type=int)
    parser.add_argument("--ckpt-every", default=1, type=int)
    parser.add_argument("--patch-size", default=8, type=int)
    parser.add_argument("--qk-dim", default=64, type=int)
    parser.add_argument("--mask-ratio", default=0.85, type=float)
    parser.add_argument("--blocks", default=1, type=int)
    parser.add_argument("--out-dim", default=None, type=int)
    parser.add_argument("--tkn-dim", default=256, type=int, help="token dimension")
    parser.add_argument("--nheads", default=12, type=int)
    parser.add_argument("--attn-beta", default=None, type=float)
    parser.add_argument("--hn-mult", default=4.0, type=float)
    parser.add_argument(
        "--alpha", default=1.0, type=float, help="step size for ET's dynamic"
    )
    parser.add_argument("--attn-bias", default=False, type=str2bool)
    parser.add_argument("--hn-bias", default=False, type=str2bool)
    parser.add_argument(
        "--time-steps", default=12, type=int, help="number of timesteps for ET"
    )
    parser.add_argument("--result-dir", default="./results", type=str)
    parser.add_argument("--num-workers", default=0, type=int)
    parser.add_argument("--batch-size", default=128, type=int)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--lr", default=8e-5, type=float, help="learning rate")
    parser.add_argument("--b1", default=0.9, type=float, help="adam beta1")
    parser.add_argument("--b2", default=0.999, type=float, help="adam beta2")

    parser.add_argument(
        "--avg-gpu",
        default=True,
        type=str2bool,
        help="a flag indicating to divide loss by the number of devices",
    )

    parser.add_argument(
        "--weight-decay", default=0.001, type=float, help="weight decay value"
    )

    parser.add_argument(
        "--data-path", default="./", type=str, help="root folder of dataset"
    )

    parser.add_argument(
        "--data-name", default="cifar10", type=str, choices=["cifar10", "cifar100"]
    )
    args = parser.parse_args()
    main(args)



