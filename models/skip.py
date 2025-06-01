import torch
import torch.nn as nn
from models.common import *

def create_skip_connection_model(
        input_channels=2,
        output_channels=3,
        down_channels=[16, 32, 64, 128, 128],
        up_channels=[16, 32, 64, 128, 128],
        skip_channels=[4, 4, 4, 4, 4],
        down_filter_size=3,
        up_filter_size=3,
        skip_filter_size=1,
        use_sigmoid=True,
        use_bias=True,
        padding='zero',
        upsampling_method='nearest',
        downsampling_method='stride',
        activation_function='LeakyReLU',
        include_1x1_up=True
    ):

    """ Creates an encoder-decoder model with skip connections.

    Arguments:
        activation_function: Either a string ('LeakyReLU', 'Swish', 'ELU', 'none') or a module (like nn.ReLU)
        padding (string): Options are 'zero' or 'reflection' (default: 'zero')
        upsampling_method (string): Options are 'nearest' or 'bilinear' (default: 'nearest')
        downsampling_method (string): Options include 'stride', 'avg', 'max', 'lanczos2' (default: 'stride')
    """

    """
    Creates an encoder-decoder model with skip connections.

    Arguments:
        input_channels (int):
            number of input channels for the network (e.g., 3 for RGB images).
        output_channels (int):
            number of output channels for the network (e.g., 3 for RGB images).
        down_channels (int[]):
            a list of the number of channels in each layer of the encoder.
        up_channels (int[]):
            a list of the number of channels in each layer of the decoder.
        skip_channels (int[]):
            a list of the number of channels in each skip connection.
        down_filter_size (int): 
            filter size for the encoder convolutional layers.
        up_filter_size (int):
            filter size for the decoder convolutional layers.
        skip_filter_size (int):
            filter size for the skip convolutional layers.
        use_sigmoid (bool):
            apply a sigmoid activation function to the output.
        use_bias (bool):
            add a bias term in the convolutional layers.
        padding (str):
            type of padding in convolutional layers. 'zero' | 'reflection'.
        upsampling_method (str or str[]):
            the method to use for upsampling in the decoder. 'nearest' | 'bilinear'.
        downsampling_method (str): 
            the method to use for downsampling in the encoder. 'stride' | 'avg' | 'max' | 'lanczos2'.
        activation_function (str): 
            the activation function to use in the network.
        include_1x1_up (bool): 
            include additional 1x1 convolution layers in the decoder path.

    Returns:
        nn.Module: 
            A PyTorch neural network model.
    """

    # check parameters and initialize necessary variables

    if not (len(down_channels) == len(up_channels) == len(skip_channels)):
        raise ValueError("The lengths of down_channels, up_channels, and skip_channels must be equal.")

    num_scales = len(down_channels)

    if not isinstance(upsampling_method, (list, tuple)):
        upsampling_method = [upsampling_method] * num_scales

    if not isinstance(downsampling_method, (list, tuple)):
        downsampling_method = [downsampling_method] * num_scales

    if not isinstance(down_filter_size, (list, tuple)):
        down_filter_size = [down_filter_size] * num_scales

    if not isinstance(up_filter_size, (list, tuple)):
        up_filter_size = [up_filter_size] * num_scales

    # Model Construction
    model = nn.Sequential()
    build_model(model, input_channels, down_channels, up_channels, skip_channels, down_filter_size, up_filter_size,
                skip_filter_size, use_bias, padding, upsampling_method, downsampling_method, activation_function, include_1x1_up)
    add_output_layer(model, output_channels, up_channels, use_sigmoid, use_bias, padding)

    return model

    
# Add Model Layers
def build_model(model, 
        input_channels=2,
        down_channels=[16, 32, 64, 128, 128],
        up_channels=[16, 32, 64, 128, 128],
        skip_channels=[4, 4, 4, 4, 4],
        down_filter_size=3,
        up_filter_size=3,
        skip_filter_size=1,
        use_bias=True,
        padding='zero',
        upsampling_method='nearest',
        downsampling_method='stride',
        activation_function='LeakyReLU',
        include_1x1_up=True
    ):
    
    num_scales = len(down_channels)
    last_index = num_scales - 1

    temp_model = model
    current_depth = input_channels

    for scale in range(num_scales):
        down_layer = nn.Sequential()
        skip_layer = nn.Sequential()

        if skip_channels[scale] > 0:
            temp_model.add(Concat(1, skip_layer, down_layer))
        else:
            temp_model.add(down_layer)

        temp_model.add(bn(skip_channels[scale] + (up_channels[scale + 1] if scale < last_index else down_channels[scale])))

        if skip_channels[scale] > 0:
            skip_layer.add(conv(current_depth, skip_channels[scale], skip_filter_size, bias=use_bias, pad=padding))
            skip_layer.add(bn(skip_channels[scale]))
            skip_layer.add(act(activation_function))


        down_layer.add(conv(current_depth, down_channels[scale], down_filter_size[scale], stride=2, bias=use_bias, pad=padding,downsample_mode=downsampling_method[scale]))
        down_layer.add(bn(down_channels[scale]))
        down_layer.add(act(activation_function))

        down_layer.add(conv(down_channels[scale], down_channels[scale], down_filter_size[scale], bias=use_bias, pad=padding))
        down_layer.add(bn(down_channels[scale]))
        down_layer.add(act(activation_function))

        deeper_branch = nn.Sequential()

        if scale == num_scales - 1:
            next_channel_count = down_channels[scale]
        else:
            down_layer.add(deeper_branch)
            next_channel_count = up_channels[scale + 1]
            
        down_layer.add(nn.Upsample(scale_factor=2, mode=upsampling_method[scale]))

        temp_model.add(conv(skip_channels[scale] + next_channel_count, up_channels[scale], up_filter_size[scale], bias=use_bias, pad=padding))
        temp_model.add(bn(up_channels[scale]))
        temp_model.add(act(activation_function))

        if include_1x1_up:
            temp_model.add(conv(up_channels[scale], up_channels[scale], 1, bias=use_bias, pad=padding))
            temp_model.add(bn(up_channels[scale]))
            temp_model.add(act(activation_function))

        current_depth = down_channels[scale]
        temp_model = deeper_branch

# Final Output Layer
def add_output_layer(model, 
        output_channels=3,
        up_channels=[16, 32, 64, 128, 128],
        use_sigmoid=True,
        use_bias=True,
        padding='zero',
    ):
    # Add the final output layer
    model.add(conv(up_channels[0], output_channels, 1, bias=use_bias, pad=padding))

    if use_sigmoid:
        model.add(nn.Sigmoid())
