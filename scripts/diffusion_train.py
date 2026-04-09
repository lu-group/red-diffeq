import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from red_diffeq.models.diffusion import Trainer, Unet, GaussianDiffusion
model = Unet(
    dim = 64,
    dim_mults = (1, 2, 4, 8),
    flash_attn = False,
    channels = 1
)

diffusion = GaussianDiffusion(
    model,
    image_size = 72,
    timesteps = 1000,    # number of steps
    sampling_timesteps = 250,
    objective = 'pred_noise'
)

training_images = []
# There are 60 sets of dataset for CurveVel-b and FlatVel-b, we take 90% for training
# That is 60 * 0.9 = 54
num_dataset_loaded = 0
for i in range(1, 55):
    training_images.append(np.load(f'dataset/Velocity_Data/CurveVel_b/model{i}.npy'))
    num_dataset_loaded += 1
    if i == 54:
        print(f'Last dataset loaded is CurveVel-b/model{i}.npy')
        print(f'Number of Dataset Loaded: {num_dataset_loaded}')
        
for i in range(1, 55):
    training_images.append(np.load(f'dataset/Velocity_Data/FlatVel_b/model{i}.npy'))
    num_dataset_loaded += 1
    if i == 54:
        print(f'Last dataset loaded is FlatVel-b/model{i}.npy')
        print(f'Number of Dataset Loaded: {num_dataset_loaded}')
    
for i in [6, 7, 8]:
    for j in range(18):
        training_images.append(np.load(f'dataset/Velocity_Data/CurveFault_b/vel{i}_1_{j}.npy'))
        num_dataset_loaded += 1
        print(f'Last dataset loaded is CurveFault-b/vel{i}_1_{j}.npy')
        if j == 17:
            print(f'Last dataset loaded is CurveFault-b/vel{i}_1_{j}.npy')
            print(f'Number of Dataset Loaded: {num_dataset_loaded}')
            break

for i in [6, 7, 8]:
    for j in range(18):
        training_images.append(np.load(f'dataset/Velocity_Data/FlatFault_b/vel{i}_1_{j}.npy'))
        num_dataset_loaded += 1
        if j == 17:
            print(f'Last dataset loaded is FlatFault-b/vel{i}_1_{j}.npy')
            print(f'Number of Dataset Loaded: {num_dataset_loaded}')
            break
        
training_images = np.concatenate(training_images)
training_images = (training_images-1500)/3000
training_images = torch.as_tensor(training_images) # images are normalized from 0 to 1
training_images = F.pad(training_images,(1,1,1,1),  "constant", 0)
print(training_images.shape)

trainer = Trainer(
    diffusion,
    training_images,
    train_batch_size =32,
    train_lr = 0.0002,
    train_num_steps = 400000,         # total training steps
    gradient_accumulate_every = 1,    # gradient accumulation steps
    ema_decay = 0.995,                # exponential moving average decay
    amp = True,                       # turn on mixed precision
    calculate_fid = False,              # whether to calculate fid during training
    results_folder = './Diffusion_checkpoint_balanced'
)

trainer.train()
