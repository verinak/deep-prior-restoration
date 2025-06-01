from models.skip import create_skip_connection_model
from utils.common_utils import *
from utils.model_utils import *

import optuna
import torch
import torch.optim
from torch.optim import Adam
from torch.nn import MSELoss
from skimage.metrics import peak_signal_noise_ratio

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark =True
dtype = torch.cuda.FloatTensor

class InpaintingModel:
    def __init__(self,
                n_iter=3500,
                input_depth=2,
                input_type = 'meshgrid',
                noise_sd = 0.03,
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
        # # Load images
        # self.img_pil = img_pil
        # self.img_np = pil_to_np(img_pil)
        # self.img_mask_pil = mask_pil
        # self.img_mask_np = pil_to_np(mask_pil)
        # self.img_mask_pil = crop_image(self.img_mask_pil, dim_div_by)
        # self.img_pil = crop_image(self.img_pil, dim_div_by)
        # self.img_np = pil_to_np(self.img_pil)
        # self.img_mask_np = pil_to_np(self.img_mask_pil)
        # self.img_mask_var = np_to_torch(self.img_mask_np).type(self.dtype)

    def setup_network(self, input_depth):
        net = create_skip_connection_model(
            input_channels = input_depth,
            output_channels = 3, # 7asa eni bal3ab f 7aga m4 el mafroud al3ab fiha
            down_channels=[128] * 5,
            up_channels=[128] * 5,
            skip_channels=[16, 32, 64, 128, 128],
            up_filter_size=3,
            down_filter_size=3,
            upsampling_method='nearest',
            skip_filter_size=1,
            use_sigmoid=True,
            use_bias=True,
            padding='reflection',
            activation_function='LeakyReLU')
        net = net.to(self.device) # move to gpu
        return net
    

    def fit(self, img_np, mask_np):
        # get netwok
        net = self.net
        
        # generate input noise (z)
        net_input = get_noise(self.input_depth, self.input_type, img_np.shape[1:])
        net_input  = net_input.to(self.device).detach()  # move to gpu, detach == don't track gradient

        # set loss and optimizer
        mse = MSELoss().to(self.device)  # move to gpu
        optimizer = Adam(net.parameters(), lr=self.learning_rate)

        # convert image to tensor
        img_var = np_to_torch(img_np).to(self.device)
        mask_var = np_to_torch(mask_np).to(self.device)
        #masked_img_var = img_var * mask_var

        # initialize variables to be used in optimization
        initial_net_input = net_input.detach().clone()
        noise = net_input.detach().clone()

        last_params = None
        last_psnr_noisy = 0
        # optimization loop
        for i in range(self.n_iter):
            def closure():
                nonlocal last_params, last_psnr_noisy
                
                optimizer.zero_grad() # reset all gradiets to zero

                # add regularization noise
                if self.noise_sd > 0:
                    noise.normal_() # generate new normal noise
                    net_input = initial_net_input + (noise * self.noise_sd)
                
                # run z through network
                net_input = net_input.float() # m3raf4 leh bs bte2leb double fl meshgrid?
                output = net(net_input)
                
                # calculate loss
                total_loss = mse(output * mask_var, img_var * mask_var)
                # optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                # calculate psnr
                output_np = torch_to_np(output)
                psnr_noisy = peak_signal_noise_ratio(img_np, output_np)

                if self.print_iter and i % self.print_iter == 0:
                    print(f'Iteration {i}   Loss {total_loss}   PSNR_noisy {psnr_noisy}')
                
                # Backtracking
                if (i+1) % 100 == 0:
                    # if psnr has dropped by 5
                    if psnr_noisy - last_psnr_noisy < -5:
                        # revert to last saved parameters
                        print('backtracking..')
                        for new_param, net_param in zip(last_params, net.parameters()):
                            net_param.data.copy_(new_param.cuda())
                        params = [x.detach().cpu() for x in net.parameters()]
                        # stop iteration
                        return total_loss
                    
                    # Store current parameters
                    last_params = [x.detach().cpu() for x in net.parameters()]
                    last_psnr_noisy = psnr_noisy


                return total_loss
            
            loss = optimizer.step(closure)

        net_input = net_input.float() # same??
        output = net(net_input)
        return output

    def inpaint(self, img_pil, mask_pil, return_all=False):
        # load image and mask
        img_pil = crop_image(img_pil)
        img_pil = ensure_rgb(img_pil)
        img_np = pil_to_np(img_pil)
        mask_pil = crop_image(mask_pil)
        mask_pil = ensure_rgb(mask_pil)
        mask_np = pil_to_np(mask_pil)

        output = self.fit(img_np=img_np, mask_np=mask_np)
        output_np = torch_to_np(output)
        result = {
            'img_cropped_pil': img_pil,
            'img_cropped_np': img_np,
            'mask_cropped_pil': mask_pil,
            'mask_cropped_np': mask_np,
            'output_np': output_np,
            'output_pil': np_to_pil(output_np),
        }
        
        if return_all:
            return result
        else:
            return result['output_pil']

