import os
import torch
import argparse
import numpy as np
from einops import reduce
from torch.utils.data import DataLoader
from torchvision.transforms import Resize
from torchvision.utils import save_image
from time import time
from accelerate import Accelerator

from image_et import (
    Patch,
    GetCIFAR,
    gen_mask_id,
    count_parameters,
    str2bool,
    get_latest_file,
)
from image_et.dynamic_et import ET as DynamicET


def make_dir(path: str):
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)


def main(args):
    IMAGE_FOLDER = os.path.join(args.result_dir, "images")
    MODEL_FOLDER = os.path.join(args.result_dir, "models")

    accelerator = Accelerator()
    device = accelerator.device

    if accelerator.is_main_process:
        make_dir(args.result_dir)
        make_dir(IMAGE_FOLDER)
        make_dir(MODEL_FOLDER)

    # model + patcher
    x = torch.randn(1, 3, 32, 32)
    patch_fn = Patch(dim=args.patch_size)
    model = DynamicET(
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
        alpha_init=args.alpha,
    )

    if accelerator.is_main_process:
        print(f"Number of parameters: {count_parameters(model)}", flush=True)

    num_patches = model.patch.N
    num_masks = int(model.patch.N * args.mask_ratio)
    trainset, testset, unnormalize_fn = GetCIFAR(args.data_path, args.data_name)

    train_loader, test_loader = map(
        lambda ds: DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            drop_last=True,
            pin_memory=True,
        ),
        (trainset, testset),
    )

    opt = torch.optim.AdamW(
        model.parameters(), lr=args.lr, betas=(args.b1, args.b2), weight_decay=args.weight_decay
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        opt,
        T_0=10,
        T_mult=1,
        eta_min=1e-6,
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
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        running_loss = 0.0
        start_time = time()

        for x, _ in train_loader:
            B = x.size(0)

            # mask indices and boolean mask for model
            batch_id, mask_id = gen_mask_id(num_patches, num_masks, B)
            bool_mask = torch.zeros(B, num_patches, dtype=torch.bool)
            bool_mask[batch_id, mask_id] = True

            # targets: original tokens at masked positions
            patched_x = patch_fn(x)  # [B, N, D]
            y = patched_x[batch_id, mask_id]  # [B, M, D]

            x, y, bool_mask = map(lambda z: z.to(device), (x, y, bool_mask))

            # use learned per-block alphas (alpha=None)
            pred = model(x, mask=bool_mask, alpha=None)

            # recovered masked tokens
            yh = pred[batch_id.to(device), mask_id.to(device)]

            loss = reduce((yh - y) ** 2, "b ... -> b", "mean").mean()
            accelerator.backward(loss)

            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), 1.0)

            opt.step()
            opt.zero_grad()
            running_loss += loss.item()

        torch.cuda.synchronize() if torch.cuda.is_available() else None
        epoch_time = time() - start_time
        scheduler.step()

        if accelerator.is_main_process:
            avg_loss = torch.tensor(running_loss / len(train_loader), device=device)
            avg_loss = avg_loss / accelerator.num_processes
            print(
                f"Epoch: {epoch}/{args.epochs}, Loss: {avg_loss:.6f}, Time: {epoch_time:.5f}s",
                flush=True,
            )

        if epoch % args.ckpt_every == 0:
            if accelerator.is_main_process:
                with torch.no_grad():
                    # visualize reconstruction of the last batch
                    x_cpu, pred_cpu = x.detach().cpu(), pred.detach().cpu()
                    bool_mask_cpu = bool_mask.detach().cpu()
                    x_masked = patch_fn(x_cpu)
                    x_masked[bool_mask_cpu] = 0.0
                    x_vis, x_masked_vis, pred_vis = map(lambda z: z[:visual_num], (x_cpu, x_masked, pred_cpu))
                    x_masked_vis, pred_vis = map(lambda z: patch_fn(z, reverse=True), (x_masked_vis, pred_vis))
                    img = Resize((64, 64), antialias=True)(torch.cat([x_vis, x_masked_vis, pred_vis], dim=0))
                    img = unnormalize_fn(img)
                    save_image(
                        img,
                        os.path.join(IMAGE_FOLDER, f"{epoch}.png"),
                        nrow=4,
                        normalize=True,
                        scale_each=True,
                    )

                try:
                    ckpt = {
                        "epoch": epoch + 1,
                        "model": model.module.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "opt": opt.state_dict(),
                        "args": args,
                    }
                except Exception:
                    ckpt = {
                        "epoch": epoch + 1,
                        "model": model.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "opt": opt.state_dict(),
                        "args": args,
                    }
                torch.save(ckpt, os.path.join(MODEL_FOLDER, f"{epoch}.pth"))

            accelerator.wait_for_everyone()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Dynamic ET with trainable alphas")
    parser.add_argument("--global-seed", default=3407, type=int)
    parser.add_argument("--ckpt-every", default=1, type=int)
    parser.add_argument("--patch-size", default=4, type=int)
    parser.add_argument("--qk-dim", default=64, type=int)
    parser.add_argument("--mask-ratio", default=0.85, type=float)
    parser.add_argument("--blocks", default=1, type=int)
    parser.add_argument("--out-dim", default=None, type=int)
    parser.add_argument("--tkn-dim", default=256, type=int, help="token dimension")
    parser.add_argument("--nheads", default=12, type=int)
    parser.add_argument("--attn-beta", default=None, type=float)
    parser.add_argument("--hn-mult", default=4.0, type=float)
    parser.add_argument(
        "--alpha", default=1.0, type=float, help="initial step size for each ET block"
    )
    parser.add_argument("--attn-bias", default=False, type=str2bool)
    parser.add_argument("--hn-bias", default=False, type=str2bool)
    parser.add_argument(
        "--time-steps", default=12, type=int, help="number of timesteps for ET"
    )
    parser.add_argument("--result-dir", default="./results_dynamic", type=str)
    parser.add_argument("--num-workers", default=0, type=int)
    parser.add_argument("--batch-size", default=128, type=int)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--lr", default=8e-5, type=float, help="learning rate")
    parser.add_argument("--b1", default=0.9, type=float, help="adam beta1")
    parser.add_argument("--b2", default=0.999, type=float, help="adam beta2")
    parser.add_argument("--avg-gpu", default=True, type=str2bool)
    parser.add_argument("--weight-decay", default=0.001, type=float)
    parser.add_argument("--data-path", default="./", type=str)
    parser.add_argument("--data-name", default="cifar10", type=str, choices=["cifar10", "cifar100"])

    args = parser.parse_args()
    main(args)

