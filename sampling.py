import os
import torch
import argparse
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from modules.dataset import SpectrumDataset
from modules.model import UNETv3
import modules.guided_diffusion as gd

def main():
    args = get_args()
    data_folder = Path(args.data_folder)
    test_folder = data_folder/'train'

    # Getting data
    data = SpectrumDataset(test_folder)
    device = torch.device("cuda:0" if torch.cuda.is_available() else torch.device('cpu'))
    BATCH_SIZE = 4
    dataloader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=False)
    (x_sample,y_sample) = next(iter(dataloader))

    # Loading model
    weights_dir = Path(os.getcwd())/'weights'/'v3'
    training_epochs = 30
    device = torch.device("cuda:0" if torch.cuda.is_available() else torch.device('cpu'))
    model = UNETv3(in_channels=80,residual=False, attention_res=[]).to(device)
    model.load_state_dict(torch.load(weights_dir/f"model_{training_epochs}.pth", map_location=device))

    # Sampling
    diffusion = create_gaussian_diffusion()
    y_predicted = diffusion.p_sample_loop(model, y_sample.shape, x_sample, progress=True, clip_denoised=True)

    # Saving
    save_folder = test_folder/'predicted'
    save_folder.mkdir(parents=True, exist_ok=True)
    np.save(save_folder/'first_batch', y_predicted.detach().cpu().numpy())


def get_args():
    parser = argparse.ArgumentParser(description='Trains diffusion model')
    parser.add_argument('data_folder', type=str, help='parent folder where data is stored')
    return parser.parse_args() 

def create_gaussian_diffusion(
        *,
        steps=1000,
        learn_sigma=False,
        sigma_small=False,
        noise_schedule="linear",
        use_kl=False,
        predict_xstart=False,
        rescale_timesteps=False,
        rescale_learned_sigmas=False,
        timestep_respacing="",
):
    betas = gd.get_named_beta_schedule(noise_schedule, steps)
    if use_kl:
        loss_type = gd.LossType.RESCALED_KL
    elif rescale_learned_sigmas:
        loss_type = gd.LossType.RESCALED_MSE
    else:
        loss_type = gd.LossType.MSE
    if not timestep_respacing:
        timestep_respacing = [steps]
    return gd.SpacedDiffusion(
        use_timesteps=gd.space_timesteps(steps, timestep_respacing),
        betas=betas,
        model_mean_type=(
            gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
    )

if __name__=='__main__':
    main()