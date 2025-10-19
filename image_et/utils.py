import torch, numpy as np
from functools import partial
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100


CIFAR10_STD = (0.4914, 0.4822, 0.4465)
CIFAR10_MU = (0.2023, 0.1994, 0.2010)

CIFAR100_STD = (0.5071, 0.4867, 0.4408)
CIFAR100_MU = (0.2675, 0.2565, 0.2761)


def gen_mask_id(num_patch, mask_size, batch_size: int):
    batch_id = torch.arange(batch_size)[:, None]
    mask_id = torch.randn(batch_size, num_patch).argsort(-1)[:, :mask_size]
    return batch_id, mask_id


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def unnormalize(x, std, mean):
    x = x * std + mean
    return x


def device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    return False


def GetCIFAR(root, which: str = "cifar10"):
    which = which.lower()
    if which == "cifar10":
        std, mean = CIFAR10_STD, CIFAR10_MU

        trainset = CIFAR10(
            root,
            train=True,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(std, mean),
                    transforms.RandomHorizontalFlip(0.5),
                    transforms.RandomVerticalFlip(0.5),
                ]
            ),
        )

        testset = CIFAR10(
            root,
            train=False,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(std, mean),
                ]
            ),
        )

    elif which == "cifar100":
        std, mean = CIFAR100_STD, CIFAR100_MU

        trainset = CIFAR100(
            root,
            train=True,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(std, mean),
                    transforms.RandomHorizontalFlip(0.5),
                    transforms.RandomVerticalFlip(0.5),
                ]
            ),
        )

        testset = CIFAR100(
            root,
            train=False,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(std, mean),
                ]
            ),
        )
    else:
        raise NotImplementedError("Not Available.")

    std, mean = map(lambda z: np.array(z)[None, :, None, None], (std, mean))

    return (trainset, testset, partial(unnormalize, std=std, mean=mean))


def wavelet_reparameterization(x: torch.Tensor, wavelet_type: str = 'haar') -> torch.Tensor:
    """
    Performs Haar wavelet transformation on each 2x2 block of the input images and
    stores the coefficients in a new tensor `wavelet_x`.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).
        wavelet_type (str, optional): Type of wavelet to use. Currently supports 'haar'.
                                      Defaults to 'haar'.

    Returns:
        torch.Tensor: Tensor of the same shape as `x`, containing wavelet coefficients.
                      Each 2x2 block in `wavelet_x` contains:
                        - Top-Left: Approximation coefficient
                        - Top-Right: Horizontal detail coefficient
                        - Bottom-Right: Vertical detail coefficient
                        - Bottom-Left: Diagonal detail coefficient

    Example:
        >>> batch_size, channels, height, width = 2, 3, 4, 4
        >>> x = torch.arange(batch_size * channels * height * width, dtype=torch.float32).reshape(batch_size, channels, height, width)
        >>> wavelet_x = wavelet_reparameterization(x)
        >>> print(wavelet_x.shape)  # torch.Size([2, 3, 4, 4])
    """
    if wavelet_type != 'haar':
        raise NotImplementedError(f"Wavelet type '{wavelet_type}' is not supported.")

    batch_size, channels, height, width = x.shape

    if height % 2 != 0 or width % 2 != 0:
        raise ValueError("Height and Width of the input tensor must be even numbers.")

    # Fold the spatial dimensions into non-overlapping 2x2 blocks
    # Reshape x to (batch_size, channels, height//2, 2, width//2, 2)
    x_reshaped = x.view(batch_size, channels, height // 2, 2, width // 2, 2)

    # Permute to bring the 2x2 blocks together: (batch_size, channels, height//2, width//2, 2, 2)
    x_permuted = x_reshaped.permute(0, 1, 2, 4, 3, 5).contiguous()

    # Now, reshape to (batch_size, channels, height//2, width//2, 4)
    # where the last dimension represents the 2x2 block
    x_blocks = x_permuted.view(batch_size, channels, height // 2, width // 2, 4)

    # Define Haar wavelet transform matrix
    haar_matrix = torch.tensor([[1, 1, 1, 1],
                                [1, -1, 1, -1],
                                [1, 1, -1, -1],
                                [1, -1, -1, 1]], dtype=x.dtype, device=x.device) / 2.0

    # Perform matrix multiplication to get wavelet coefficients
    # x_blocks shape: (batch_size, channels, height//2, width//2, 4)
    # haar_matrix shape: (4, 4)
    # Output shape: same as x_blocks
    wavelet_blocks = torch.matmul(x_blocks, haar_matrix.T)

    # Assign the coefficients to the corresponding positions in wavelet_x
    # Initialize wavelet_x with zeros
    wavelet_x = torch.zeros_like(x)

    # Map the coefficients back to spatial positions
    # wavelet_blocks has shape (batch_size, channels, height//2, width//2, 4)
    # We need to map it back to (batch_size, channels, height, width)
    # with the corresponding coefficients in 2x2 blocks

    # Permute back to (batch_size, channels, height//2, width//2, 2, 2)
    wavelet_blocks_perm = wavelet_blocks.view(batch_size, channels, height // 2, width // 2, 2, 2)

    # Permute to (batch_size, channels, height//2, 2, width//2, 2)
    wavelet_blocks_perm = wavelet_blocks_perm.permute(0, 1, 2, 4, 3, 5).contiguous()

    # Finally, reshape to (batch_size, channels, height, width)
    wavelet_x = wavelet_blocks_perm.view(batch_size, channels, height, width)

    return wavelet_x


if __name__ == "__main__":
    x = torch.tensor([[[[0.0, 1.0], 
                        [1.0, 0.0]]]], dtype=torch.float32)
    print(x.shape)
    wavelet_x = wavelet_reparameterization(x)
    print(wavelet_x)
