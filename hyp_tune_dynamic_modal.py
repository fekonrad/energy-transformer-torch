import os
import subprocess
from typing import Optional

import modal


app = modal.App("energy-transformer-dynamic-tune")


# Modal image with the project dependencies. Mirrors requirements.txt.
image = (
    modal.Image.debian_slim()
    .pip_install(
        "torch",
        "torchvision",
        "numpy<2.0",
        "pillow>=9.5",
        "matplotlib",
        "tqdm",
        "einops",
        "accelerate",
    )
    .add_local_dir("image_et", "/root/et/image_et")
    .add_local_file("train.py", "/root/et/train.py")
    .add_local_file("train_dynamic.py", "/root/et/train_dynamic.py"))

# Persist results across runs (create if missing).
results_vol = modal.Volume.from_name("et-results", create_if_missing=True)


@app.function(
    image=image,
    gpu="A10G",
    timeout=60 * 60 * 24,
    volumes={"/vol": results_vol},
)
def run_dyn_job(
    time_steps: int,
    epochs: int,
    batch_size: int,
    lr: float,
    data_name: str,
    patch_size: int,
    tkn_dim: int,
    qk_dim: int,
    nheads: int,
    hn_mult: float,
    attn_beta: float,
    attn_bias: bool,
    hn_bias: bool,
    blocks: int,
    alpha_init: float,
    mask_ratio: float,
    weight_decay: float,
    b1: float,
    b2: float,
    result_subdir: Optional[str] = None,
):
    cwd = "/root/et"
    result_dir = (
        f"/vol/dynamic_sweep/ts_{time_steps}"
        if result_subdir is None
        else f"/vol/{result_subdir}"
    )
    os.makedirs(result_dir, exist_ok=True)

    cmd = [
        "python",
        "train_dynamic.py",
        "--patch-size",
        str(patch_size),
        "--tkn-dim",
        str(tkn_dim),
        "--qk-dim",
        str(qk_dim),
        "--nheads",
        str(nheads),
        "--hn-mult",
        str(hn_mult),
        "--attn-beta",
        str(attn_beta),
        "--attn-bias",
        str(bool(attn_bias)),
        "--hn-bias",
        str(bool(hn_bias)),
        "--time-steps",
        str(time_steps),
        "--alpha",
        str(alpha_init),
        "--blocks",
        str(blocks),
        "--epochs",
        str(epochs),
        "--result-dir",
        result_dir,
        "--batch-size",
        str(batch_size),
        "--lr",
        str(lr),
        "--b1",
        str(b1),
        "--b2",
        str(b2),
        "--mask-ratio",
        str(mask_ratio),
        "--weight-decay",
        str(weight_decay),
        "--data-path",
        cwd,
        "--data-name",
        data_name,
    ]

    print(f"== Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=cwd)
    return {"time_steps": time_steps, "result_dir": result_dir}


@app.local_entrypoint()
def main(
    # Sweep space (time_steps only)
    # time_steps_list = [1, 2, 4, 8, 16, 32],
    time_steps_list = [1],
    # Training knobs (align with hpc/run_dynamic.slurm defaults)
    epochs: int = 64,
    batch_size: int = 128,
    lr: float = 8e-4,
    data_name: str = "cifar10",
    # Model knobs
    patch_size: int = 4,
    tkn_dim: int = 256,
    qk_dim: int = 64,
    nheads: int = 12,
    hn_mult: float = 4.0,
    attn_beta: float = 0.125,
    attn_bias: bool = False,
    hn_bias: bool = False,
    blocks: int = 4,
    alpha_init: float = 1.0,
    mask_ratio: float = 0.50,
    weight_decay: float = 0.0001,
    b1: float = 0.9,
    b2: float = 0.999,
):
    futures = []
    for ts in time_steps_list:
        subdir = f"dynamic_sweep/ts_{ts}"
        print(f"== Scheduling time_steps={ts} -> /vol/{subdir}")
        fut = run_dyn_job.spawn(
            ts,
            epochs,
            batch_size,
            lr,
            data_name,
            patch_size,
            tkn_dim,
            qk_dim,
            nheads,
            hn_mult,
            attn_beta,
            attn_bias,
            hn_bias,
            blocks,
            alpha_init,
            mask_ratio,
            weight_decay,
            b1,
            b2,
            subdir,
        )
        futures.append(fut)

    results = [f.get() for f in futures]
    print("== Completed jobs:")
    for r in results:
        print(r)
