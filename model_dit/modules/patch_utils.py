import torch
from einops import rearrange


def unpatchify(hidden_states, patch_size, out_channels, input_batch_size):
    """
    Reconstruct the original image from patchified hidden states.

    Args:
        hidden_states (torch.Tensor): The patchified tensor with shape
                                      (batch_size * num_frames, num_patches, patch_dim).
        patch_size (int): The size of the patches (assumed square).
        out_channels (int): The number of output channels (e.g., 3 for RGB images).
        input_batch_size (int): The original batch size before any reshaping for frames.

    Returns:
        torch.Tensor: The reconstructed tensor with shape
                      (batch_size, out_channels, num_frames, height, width).
    """

    # Step 1: Calculate height and width from the number of patches (assuming square input).
    # We assume that the second dimension of hidden_states represents the number of patches.
    num_patches = hidden_states.shape[1]
    height = width = int(num_patches**0.5)  # Assuming the patches form a square grid.

    # Step 2: Reshape the hidden states to include patch spatial dimensions (height, width, patch_size, patch_size, out_channels).
    hidden_states = hidden_states.reshape(shape=(-1, height, width, patch_size, patch_size, out_channels))

    # Step 3: Use einsum to reorder dimensions for better patch layout.
    # This converts the shape (batch, height, width, patch_size, patch_size, channels)
    # to (batch, channels, height, patch_size, width, patch_size)
    hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states)

    # Step 4: Reshape to combine patch dimensions back to original spatial dimensions.
    # The final output shape will be (batch, out_channels, height * patch_size, width * patch_size)
    output = hidden_states.reshape(shape=(-1, out_channels, height * patch_size, width * patch_size))

    # Step 5: Reorganize the batch and frames dimensions back to their original form using rearrange.
    # This reshapes the tensor from (batch_size * num_frames, channels, height, width) to
    # (batch_size, channels, num_frames, height, width), where `b` is input_batch_size and `f` is num_frames.
    output = rearrange(output, "(b f) c h w -> b c f h w", b=input_batch_size).contiguous()

    return output
