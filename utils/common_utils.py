import torch
import torchvision
import numpy as np
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


def load_image(path):
    image = Image.open(path)
    return image 
 

def pil_to_np(pil_image):
    """
    Convert a PIL image to a normalized NumPy array.
    
    Args:
        pil_image: Input image in PIL format.
    
    Returns:
        NumPy array of shape (C, H, W) for color images or (1, H, W) for grayscale,
        with pixel values normalized to the range [0, 1].
    """
    
    # Convert PIL image to NumPy array
    np_image = np.array(pil_image)
    
    # Handle color images (3D array) by rearranging to (C, H, W)
    if np_image.ndim == 3:
        np_image = np_image.transpose(2, 0, 1)
    
    # Handle grayscale images (2D array) by adding a channel dimension
    else:
        np_image = np.expand_dims(np_image, axis=0)
    
    # Normalize pixel values to the range [0, 1]
    np_image = np_image.astype(np.float32) / 255
    
    return np_image


def np_to_pil(image_np):
    """
    Converts a numpy array image to a PIL image format.
    
    Transforms the image from [C x W x H] in the range [0, 1] 
    to [W x H x C] in the range [0, 255].
    """
    # Scale pixel values and clip to valid range
    array = np.clip(image_np * 255, 0, 255).astype(np.uint8)
    
    # Handle single-channel (grayscale) images
    if image_np.shape[0] == 1:
        array = array[0]  # Remove the channel dimension
    else:
        array = array.transpose(1, 2, 0)  # Reorder dimensions for PIL compatibility

    return Image.fromarray(array)


def np_to_torch(image_np):
    """
    Converts a numpy array image to a PyTorch tensor.
    
    Keeps the format as [C x W x H] in the range [0, 1].
    """
    return torch.from_numpy(image_np).unsqueeze(0)  # Add batch dimension


def torch_to_np(image_tensor):
    """
    Converts a PyTorch tensor image to a numpy array.
    
    Transforms the image from [1 x C x W x H] to [C x W x H] in the range [0, 1].
    """
    return image_tensor.detach().cpu().numpy().squeeze(0)  # Remove batch dimension


def get_image(image_path, target_size=-1):
    """
    Load an image and resize it if a target size is specified.
    
    Args:
        image_path: The file path to the image.
        target_size: Tuple or scalar for target dimensions; 
                     if -1, no resizing is applied.
    
    Returns:
        image: The loaded (and resized, if applicable) PIL Image object.
    """
    
    # Step 1: Load the image
    image = Image.open(image_path)
    
    # Step 2: Handle target_size input (convert scalar to tuple if necessary)
    if isinstance(target_size, int):
        target_size = (target_size, target_size)
    
    # Step 3: Check if resizing is needed
    should_resize = target_size[0] != -1 and target_size != image.size
    
    if should_resize:
        # Step 4: Choose appropriate resizing method based on size comparison
        resample_method = Image.BICUBIC if target_size[0] > image.size[0] else Image.ANTIALIAS
        
        # Step 5: Resize the image
        image = image.resize(target_size, resample_method)
    
    # Convert the image to a NumPy array if needed (can add the code here)
    image_np = pil_to_np(image)

    # Step 6: Return the image
    return image,image_np 


def crop_image(image, divisor=32):
    """
    Crops the image so that its dimensions are divisible by the given divisor.
    
    Args:
        image: A PIL Image object to be cropped.
        divisor: The number by which the image dimensions should be divisible (default is 32).
    
    Returns:
        Cropped image with dimensions that are multiples of the divisor.
    """
    
    # Calculate new dimensions that are divisible by the divisor
    width, height = image.size
    new_width = width - (width % divisor)
    new_height = height - (height % divisor)
    
    # Determine the crop box to center the crop
    left = (width - new_width) // 2
    top = (height - new_height) // 2
    right = left + new_width
    bottom = top + new_height

    # Crop the image using the calculated box
    cropped_image = image.crop((left, top, right, bottom))
    
    return cropped_image

def ensure_rgb(image):
    """
    Make sure image mode is RGB.

    Args:
        image: a PIL image.

    Returns:
        The image in RGB mode.
    """
    if image.mode != 'RGB':
        return image.convert('RGB')
    return image


def ensure_binary(image):
    """
    Make sure image mode is Binary.

    Args:
        image: a PIL image.

    Returns:
        The image in Binary mode.
    """
    if image.mode != '1':
        # Convert to greyscale first if necessary
        if image.mode != 'L':
            image = image.convert('L')
        return image.convert('1')
    return image

def get_image_grid(images_array, images_per_row=8):
    """Construct a grid of images from a list of NumPy arrays.

    Args:
        images_array: A list of images as NumPy arrays.
        images_per_row: The number of images to display per row.

    Returns:
        A NumPy array representing the image grid.
    """
    # Check if images_array is a list and contains valid NumPy arrays
    if not isinstance(images_array, list) or not all(isinstance(img, np.ndarray) for img in images_array):
        raise ValueError("images_array must be a list of NumPy arrays.")

    # Convert each NumPy image to a PyTorch tensor
    tensors = [torch.from_numpy(image) for image in images_array]

    # Generate a grid of images using torchvision utilities
    image_grid_tensor = torchvision.utils.make_grid(tensors, nrow=images_per_row)

    # Convert the grid tensor back to a NumPy array
    image_grid_array = image_grid_tensor.numpy()

    return image_grid_array


def plot_image_grid(images_np, images_per_row =8, size_factor=1, interp_method='lanczos'):
    """Render a grid of images using matplotlib.

    Args:
        images: A list of images, where each image is a NumPy array of shape 3xHxW or 1xHxW.
        images_per_row: The number of images to display in each row of the grid.
        size_factor: A scaling factor for the figure size.
        interp_method: The interpolation method to use in plt.imshow.
    """
    # Determine the number of channels in the images
    num_channels = max(image.shape[0] for image in images_np)
    if num_channels not in {1, 3}:
        raise ValueError("Each image must have either 1 or 3 channels.")

    # Ensure all images have the same number of channels
    images = [image if image.shape[0] == num_channels else np.concatenate([image] * 3, axis=0) for image in images_np]

    # Create a grid of images using the previously defined function
    grid_array = get_image_grid(images, nrow=images_per_row)

    # Set the figure size based on the number of images and the scaling factor
    plt.figure(figsize=(len(images) + size_factor, 12 + size_factor))

    # Display the grid of images
    if images[0].shape[0] == 1:
        plt.imshow(grid_array[0], cmap='gray', interpolation=interp_method)
    else:
        plt.imshow(grid_array.transpose(1, 2, 0), interpolation=interp_method)

    plt.axis('off')  # Turn off axis labels
    plt.show()

    return grid_array
