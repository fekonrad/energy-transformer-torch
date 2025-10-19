import os, tqdm
import torch
import argparse
import pprint
import numpy as np
from einops import reduce
from torch.utils.data.dataloader import DataLoader

from torchvision.transforms import Resize, InterpolationMode
from torchvision.utils import save_image


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

from time import time
from accelerate import Accelerator
import torch.nn.functional as F


def make_dir(dir_name: str):
    if not os.path.isdir(dir_name):
        os.mkdir(dir_name)


def haar_wavelet_transform(x):
    """
    Apply 2D Haar wavelet transform to input tensor using convolution
    Args:
        x: Input tensor of shape [batch, channels, height, width]
    Returns:
        Haar coefficients organized as [batch, channels, height, width]
        with LL (low-low) in top-left, LH in top-right, HL in bottom-left, HH in bottom-right
    """
    batch_size, channels, h, w = x.shape
    
    # Ensure dimensions are even for 2x2 blocks
    assert h % 2 == 0 and w % 2 == 0, "Height and width must be even for Haar transform"
    
    # Define Haar basis functions
    # Scaling/averaging filter (low-pass)
    h0 = torch.tensor([[1/2, 1/2], [1/2, 1/2]], device=x.device).view(1, 1, 2, 2)
    # Vertical filter (high-pass)
    h1 = torch.tensor([[1/2, 1/2], [-1/2, -1/2]], device=x.device).view(1, 1, 2, 2)
    # Horizontal filter (high-pass)
    h2 = torch.tensor([[1/2, -1/2], [1/2, -1/2]], device=x.device).view(1, 1, 2, 2)
    # Diagonal filter (high-pass)
    h3 = torch.tensor([[1/2, -1/2], [-1/2, 1/2]], device=x.device).view(1, 1, 2, 2)
    
    # Apply convolution with stride 2 to get coefficients
    # Reshape input for grouped convolution
    x_reshaped = x.view(batch_size * channels, 1, h, w)
    
    # Apply each filter
    ll = torch.nn.functional.conv2d(x_reshaped, h0, stride=2)  # Low-Low
    lh = torch.nn.functional.conv2d(x_reshaped, h1, stride=2)  # Low-High
    hl = torch.nn.functional.conv2d(x_reshaped, h2, stride=2)  # High-Low
    hh = torch.nn.functional.conv2d(x_reshaped, h3, stride=2)  # High-High
    
    # Reshape back to original batch and channel dimensions
    ll = ll.view(batch_size, channels, h//2, w//2)
    lh = lh.view(batch_size, channels, h//2, w//2)
    hl = hl.view(batch_size, channels, h//2, w//2)
    hh = hh.view(batch_size, channels, h//2, w//2)
    
    # Stack coefficients in the correct quadrants
    top = torch.cat([ll, lh], dim=3)
    bottom = torch.cat([hl, hh], dim=3)
    return torch.cat([top, bottom], dim=2)


def inverse_haar_wavelet_transform(coeffs):
    """
    Apply inverse 2D Haar wavelet transform using transposed convolution
    Args:
        coeffs: Haar coefficients organized as [batch, channels, height, width]
    Returns:
        Reconstructed image tensor
    """
    batch_size, channels, h, w = coeffs.shape
    
    # Extract quadrants
    ll = coeffs[:, :, :h//2, :w//2]  # Low-Low
    lh = coeffs[:, :, :h//2, w//2:]  # Low-High
    hl = coeffs[:, :, h//2:, :w//2]  # High-Low
    hh = coeffs[:, :, h//2:, w//2:]  # High-High
    
    # Define inverse Haar basis functions (transpose of forward transform)
    h0 = torch.tensor([[1/2, 1/2], [1/2, 1/2]], device=coeffs.device).view(1, 1, 2, 2)
    h1 = torch.tensor([[1/2, 1/2], [-1/2, -1/2]], device=coeffs.device).view(1, 1, 2, 2)
    h2 = torch.tensor([[1/2, -1/2], [1/2, -1/2]], device=coeffs.device).view(1, 1, 2, 2)
    h3 = torch.tensor([[1/2, -1/2], [-1/2, 1/2]], device=coeffs.device).view(1, 1, 2, 2)
    
    # Reshape for grouped transposed convolution
    ll = ll.view(batch_size * channels, 1, h//2, w//2)
    lh = lh.view(batch_size * channels, 1, h//2, w//2)
    hl = hl.view(batch_size * channels, 1, h//2, w//2)
    hh = hh.view(batch_size * channels, 1, h//2, w//2)
    
    # Apply transposed convolution
    x0 = torch.nn.functional.conv_transpose2d(ll, h0, stride=2)
    x1 = torch.nn.functional.conv_transpose2d(lh, h1, stride=2)
    x2 = torch.nn.functional.conv_transpose2d(hl, h2, stride=2)
    x3 = torch.nn.functional.conv_transpose2d(hh, h3, stride=2)
    
    # Sum all components and reshape back
    x = (x0 + x1 + x2 + x3).view(batch_size, channels, h, w)
    return x


def generate_superres_data(hr_images):
    """
    Generate superresolution training data using Haar wavelets
    Args:
        hr_images: High-resolution target images [batch, channels, height, width]
    Returns:
        input_data: Haar coefficients with only LL known (from downscaled image)
        target_data: Full Haar coefficients of the high-resolution image
        mask_info: Information about which coefficients to predict
    """
    batch_size, channels, h, w = hr_images.shape
    
    # Apply Haar wavelet transform to the high-resolution target
    hr_haar_coeffs = haar_wavelet_transform(hr_images)
    
    # Create input data - we know the LL coefficients from the low-res image
    # The LL coefficients of the HR image should correspond to the LR image
    input_data = torch.zeros_like(hr_haar_coeffs)
    input_data[:, :, :h//2, :w//2] = hr_haar_coeffs[:, :, :h//2, :w//2]  # Fill LL quadrant with LR image
    
    # Step 4: Target is the full Haar coefficients of the high-resolution image
    target_data = hr_haar_coeffs
    
    # Step 5: Create mask for the unknown coefficients (high-frequency components)
    mask = torch.zeros((h, w), dtype=torch.bool, device=hr_images.device)
    mask[:h//2, w//2:] = True  # LH region
    mask[h//2:, :w//2] = True  # HL region
    mask[h//2:, w//2:] = True  # HH region
    
    return input_data, target_data, mask


def main(args):
    IMAGE_FOLDER = args.result_dir + "/images"
    MODEL_FOLDER = args.result_dir + "/models"

    accelerator = Accelerator()
    device = accelerator.device

    if accelerator.is_main_process:
        make_dir(args.result_dir)
        make_dir(IMAGE_FOLDER)
        make_dir(MODEL_FOLDER)

    # ---------------------------------------------
    # 1. Load dataset first – we need its image size
    # ---------------------------------------------
    trainset, testset, unnormalize_fn = GetCIFAR(args.data_path, args.data_name)

    # Assume square images (CIFAR is 32×32)
    sample_img, _ = trainset[0]
    _, H, W = sample_img.shape
    assert H == W, "Dataset images must be square"

    patch_size = args.patch_size
    input_size = H  # e.g. 32 for CIFAR-10/100
    if input_size % patch_size != 0:
        input_size = (input_size // patch_size + 1) * patch_size

    print("\nInitialisation → dataset image side =", input_size)
    print("patch_size =", patch_size,
          "→ num patches =", (input_size // patch_size) ** 2)

    # ------------------------------------------------
    # 2. Build patcher and model using that input_size
    # ------------------------------------------------
    patch_fn = Patch(dim=patch_size, n=input_size)   # n MUST equal image side
    x_dummy  = torch.randn(1, 3, input_size, input_size)

    model = ET(
        x_dummy,
        patch_fn,
        args.out_dim if args.out_dim is not None else patch_size * patch_size * 3,
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

    # ---------------------------------------------
    # 3. Select a fixed batch from dataset for visualization
    # ---------------------------------------------
    visual_num = 16  # number of images to visualise
    viz_hr = torch.stack([testset[i][0] for i in range(visual_num)])  # shape [visual_num, C, H, W]

    # ---------------------------------------------
    # 4. Create dataloaders, optimiser, scheduler
    # ---------------------------------------------
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

    # visual_num already defined above (kept for consistency)
    training_display = range(start_epoch, args.epochs + 1)

    for i in training_display:
        running_loss = 0.0

        model.train()

        start_time = time()
        for hr_images, _ in train_loader:
            batch_size = hr_images.size(0)
            
            # Generate superresolution training data
            input_data, target_data, haar_mask = generate_superres_data(hr_images)
            
            # Convert spatial mask to patch mask
            haar_mask_float = haar_mask.float().unsqueeze(0).unsqueeze(0)
            patch_mask = patch_fn(haar_mask_float)
            patch_mask = (patch_mask.sum(dim=-1) > 0).squeeze()  # shape: [N]
            patch_mask = patch_mask.to(device)
 
            # Broadcast patch mask to full batch
            full_mask = patch_mask.unsqueeze(0).expand(batch_size, -1).clone()  # shape: [B, N]
 
            # Forward pass with the full mask
            pred = model(input_data, mask=full_mask, alpha=args.alpha)
             
            # Convert predictions back to image space
            pred_patches = patch_fn(pred, reverse=True)
             
            # Compute loss only on masked regions (high frequency components)
            # Expand spatial mask to (B, C, H, W) so it matches pred_patches
            mask_expanded = haar_mask.unsqueeze(0).unsqueeze(0).expand(batch_size, pred_patches.size(1), -1, -1)
 
            loss = F.mse_loss(
                pred_patches[mask_expanded],
                target_data[mask_expanded]
            )
            
            accelerator.backward(loss)

            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), 1.0)

            opt.step()
            opt.zero_grad()
            running_loss += loss.item()

        if device == "cuda":
            torch.cuda.synchronize()
            
        end_time = time()
        scheduler.step(running_loss)

        if accelerator.is_main_process:
            epoch_time = end_time - start_time
            avg_loss = torch.tensor(running_loss / len(train_loader), device=device)
            avg_loss = avg_loss / accelerator.num_processes
            print(
                f"Epoch: {i}/{args.epochs}, Loss: {avg_loss:.6f}, Time: {epoch_time:.5f}s",
                flush=True,
            )

        if i % args.ckpt_every == 0:
            if accelerator.is_main_process:
                with torch.no_grad():
                    # Use the fixed batch selected at startup for visualisation
                    test_hr = viz_hr.to(device)
                    test_input, test_target, test_mask = generate_superres_data(test_hr)
                    
                    # Create visualization mask
                    test_mask_float = test_mask.float().unsqueeze(0).unsqueeze(0)
                    test_patch_mask = patch_fn(test_mask_float)
                    test_patch_mask = (test_patch_mask.sum(dim=-1) > 0).squeeze().to(device)  # shape: [N]

                    # Create full mask for test batch
                    B_vis = test_input.size(0)
                    test_full_mask = test_patch_mask.unsqueeze(0).expand(B_vis, -1).clone()  # shape: [B_vis, N]
                    print(test_full_mask[0])

                    # Get predictions
                    test_pred = model(test_input.to(device), mask=test_full_mask, alpha=args.alpha)
                     
                    # Convert predictions back to image space
                    pred_image = patch_fn(test_pred, reverse=True)

                    # Move all tensors to CPU before concatenation for save_image
                    img_cpu = torch.cat([
                        test_input.cpu(),
                        test_target.cpu(),
                        pred_image.cpu()
                    ], dim=0)

                    img_cpu = unnormalize_fn(img_cpu)
 
                    save_image(
                        img_cpu,
                        IMAGE_FOLDER + "/{0}.png".format(i),
                        nrow=visual_num,
                        normalize=True,
                        scale_each=True,
                    )
                
                # Save checkpoint
                try:
                    ckpt = {
                        "epoch": i + 1,
                        "model": model.module.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "opt": opt.state_dict(),
                        "args": args,
                    }
                except:
                    ckpt = {
                        "epoch": i + 1,
                        "model": model.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "opt": opt.state_dict(),
                        "args": args,
                    }
                torch.save(ckpt, MODEL_FOLDER + f"/{i}.pth")
            accelerator.wait_for_everyone()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ET for Image Superresolution")
    parser.add_argument("--global-seed", default=3407, type=int)
    parser.add_argument("--ckpt-every", default=1, type=int)
    parser.add_argument("--patch-size", default=4, type=int)
    parser.add_argument("--qk-dim", default=64, type=int)
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

    """
    # Quick Test for data generation routine. Seems to work fine!

    import matplotlib.pyplot as plt
    x = torch.tensor([[[[2.0, -1.0], [3.0, -2.0]]]])
    x, y, mask = generate_superres_data(x)
    print(x)
    print(y)
    print(mask)
    """
