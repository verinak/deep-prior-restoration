from models.skip import create_skip_connection_model
from models.downsampler import Downsampler
from utils.common_utils import *
from utils.model_utils import *

import torch
from torch.optim import Adam
from torch.nn import MSELoss
from skimage.metrics import peak_signal_noise_ratio
from skimage.restoration import denoise_nl_means


class SuperResolutionModel:
    def __init__(self,
                n_iter=2000,
                input_depth=32,
                input_type = 'noise',
                noise_sd=0.03, # standard deviation of regularization noise
                learning_rate=0.01,
                print_iter=None,
                ):
        
        self.n_iter = n_iter
        self.input_depth = input_depth
        self.input_type = input_type
        self.noise_sd = noise_sd
        self.learning_rate = learning_rate
        self.print_iter = print_iter

        # gpu setup
        if not torch.cuda.is_available():
            raise Exception("GPU not available.")
        self.device = torch.device('cuda')

        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark =True

        # network setup
        self.net = self.setup_network(input_depth)
        

        
    
    def setup_network(self, input_depth):
        net = create_skip_connection_model(
                    input_channels = input_depth,
                    output_channels = 3,
                    down_channels = [128]*5,
                    up_channels   = [128]*5,
                    skip_channels = [4]*5,
                    upsampling_method = 'bilinear',
                    use_sigmoid = True,
                    use_bias = True,
                    padding = 'reflection',
                    activation_function = 'LeakyReLU')
        net = net.to(self.device) # move to gpu
        return net
    

    def fit(self, img_np, target_size):
        # get netwok
        net = self.net

        # generate input noise (z)
        net_input = get_noise(self.input_depth, self.input_type, target_size)
        net_input  = net_input.to(self.device).detach()  # move to gpu, detach == don't track gradient

        # set loss and optimizer
        mse = MSELoss().to(self.device)  # move to gpu
        optimizer = Adam(net.parameters(), lr=self.learning_rate)

        # convert image to torch format
        img_torch = np_to_torch(img_np).to(self.device)  # move to gpu

        # set up downsampler
        downsampler = Downsampler(n_planes=3, factor=4, kernel_type='lanczos2', phase=0.5, preserve_size=True).to(self.device)
        
        # initialize variables to be used in optimization
        initial_net_input = net_input.detach().clone()
        noise = net_input.detach().clone()
        # variables to save last parameters for backtracking
        last_params = None
        last_psnr_lr = 0
        
        # optimization loop
        for i in range(self.n_iter):
            def closure():
                nonlocal last_params, last_psnr_lr
                optimizer.zero_grad() # reset all gradiets to zero

                # add regularization noise
                if self.noise_sd > 0:
                    noise.normal_() # generate new normal noise
                    net_input = initial_net_input + (noise * self.noise_sd)

                # run z through network
                output = net(net_input)
                output_LR = downsampler(output)

                # calculate loss
                total_loss = mse(output_LR, img_torch)
                total_loss.backward()
                optimizer.step()

                # calculate psnr
                output_LR_np = torch_to_np(output_LR)
                psnr_lr = peak_signal_noise_ratio(img_np, output_LR_np)

                if self.print_iter and i % self.print_iter == 0:
                    print(f'Iteration {i}   Loss {total_loss}   PSNR_LR {psnr_lr}')

                # Backtracking
                if (i+1) % 100 == 0:
                    # if psnr has dropped by 5
                    if psnr_lr - last_psnr_lr < -5:
                        # revert to last saved parameters
                        print('backtracking..')
                        for new_param, net_param in zip(last_params, net.parameters()):
                            net_param.data.copy_(new_param.cuda())
                        # stop iteration
                        return total_loss
                    
                    # Store current parameters
                    last_params = [x.detach().cpu() for x in net.parameters()]
                    last_psnr_lr = psnr_lr
    
                
                return total_loss
            loss = optimizer.step(closure)
        
        output = net(net_input)
        return output


    def upscale(self, img_pil, return_all=False):
        # load image
        img_pil = crop_image(img_pil)
        img_pil = ensure_rgb(img_pil)
        img_np = pil_to_np(img_pil)

        new_size = (img_pil.size[0] * 4, img_pil.size[1] * 4)
        # bicubic_pil = img_pil.resize(new_size, Image.BICUBIC)
        # bicubic_np = pil_to_np(bicubic_pil)

        output = self.fit(img_np=img_np, target_size=new_size)
        output_np = torch_to_np(output)

        result = {
            'img_cropped_pil': img_pil,
            'img_cropped_np': img_np,
            'output_np': output_np,
            'output_pil': np_to_pil(output_np),
        }
        
        if return_all:
            return result
        else:
            return result['output_pil']


