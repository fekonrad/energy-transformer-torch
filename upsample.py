import torch


def upsample_wavelet(x, scale_factor=2, wavelet_type='haar'):
    """
    Upsamples the input tensor by a given scale factor using Haar wavelets.
    
    For each pixel in the input image, a 2x2 block in the upsampled image is created.
    The "constant" Haar wavelet coefficient is set to the pixel value,
    and the other coefficients are set to zero.
    
    Additionally, a binary mask is returned indicating known coefficients.
    
    Args:
        x (torch.Tensor): Input tensor of shape (b, c, h, w).
        scale_factor (int): The upsampling scale factor (e.g., 2).
        wavelet_type (str): Type of wavelet to use ('haar' supported).
    
    Returns:
        upsampled (torch.Tensor): Upsampled tensor of shape (b, c, 2h, 2w).
        mask_up (torch.Tensor): Binary mask tensor of shape (b, 1, 2h, 2w).
    """
    if wavelet_type != 'haar':
        raise NotImplementedError("Only 'haar' wavelet is supported.")
    
    if scale_factor != 2:
        raise NotImplementedError("Only a scale factor of 2 is supported.")
    
    b, c, h, w = x.size()
    
    # Create the mask: known coefficients are the father wavelet (constant)
    mask = torch.ones((b, 1, h, w), device=x.device, dtype=torch.float32)
    
    # Repeat the tensor to upsample
    upsampled = x.repeat_interleave(scale_factor, dim=2).repeat_interleave(scale_factor, dim=3)
    
    # Expand mask similarly
    mask_up = mask.repeat_interleave(scale_factor, dim=2).repeat_interleave(scale_factor, dim=3)
    
    # Zero out the unknown coefficients (detail coefficients)
    # Since only the top-left pixel in each 2x2 block contains the known coefficient
    # We need to set the other three pixels in each block to zero
    # Create a pattern mask
    pattern = torch.tensor([[1, 0],
                            [0, 0]], device=x.device, dtype=torch.bool)
    pattern = pattern.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, 2, 2)
    
    # Repeat the pattern across the batch and spatial dimensions
    pattern = pattern.repeat(b, 1, h, w)  # Shape: (b, 1, 2h, 2w)
    
    # Apply the pattern to zero out the detail coefficients
    upsampled = upsampled * pattern
    mask_up = mask_up * pattern
    mask_up = mask_up.to(torch.bool)
    
    return upsampled, mask_up


def wavelet_to_image(wavelet_coeffs):
    """
    Converts upsampled Haar wavelet coefficients back to the spatial image.

    This function takes the wavelet coefficients produced by the `upsample_wavelet`
    method and reconstructs the upscaled image by replicating the constant coefficients
    across each corresponding 2x2 block.

    Args:
        wavelet_coeffs (torch.Tensor): Wavelet coefficients tensor of shape (b, c, 2h, 2w).

    Returns:
        image_upscaled (torch.Tensor): Reconstructed upscaled image tensor of shape (b, c, 2h, 2w).
    """
    # Ensure the input tensor has the correct number of dimensions
    if wavelet_coeffs.dim() != 4:
        raise ValueError("wavelet_coeffs must be a 4D tensor of shape (b, c, 2h, 2w)")
    
    b, c, H, W = wavelet_coeffs.shape
    
    if H % 2 != 0 or W % 2 != 0:
        raise ValueError("The height and width of wavelet_coeffs must be divisible by 2")
    
    # Extract the constant (father wavelet) coefficients located at the top-left of each 2x2 block
    constant_coeffs = wavelet_coeffs[:, :, 0::2, 0::2]  # Shape: (b, c, h, w)
    
    # Replicate the constant coefficients across each 2x2 block to reconstruct the image
    image_upscaled = constant_coeffs.repeat_interleave(2, dim=2).repeat_interleave(2, dim=3)  # Shape: (b, c, 2h, 2w)
    
    return image_upscaled


if __name__ == '__main__': 
    import matplotlib.pyplot as plt 

    x = torch.tensor([[[[2, 1], [-1, 0]]]])

    wavelet_coeffs, mask = upsample_wavelet(x, 2)
    print(wavelet_coeffs.detach().numpy()[0,0])
    print(mask.detach().numpy()[0,0])

    image = wavelet_to_image(wavelet_coeffs)
    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(x.detach().numpy()[0,0])
    axs[1].imshow(image.detach().numpy()[0,0])
    plt.show()

