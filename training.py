import torch
import os
import argparse
from pathlib import Path
import numpy as np
from torch.utils.data import DataLoader
from datetime import datetime
import torch.nn.functional as func
import modules.guided_diffusion as gd
from modules.dataset import SpectrumDataset
from modules.model import UNETv1, UNETv2

def main():
    # network hyperparameters
    device = torch.device("cuda:0" if torch.cuda.is_available() else torch.device('cpu'))
    save_dir = Path(os.getcwd())/'weights'/'v2'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    args = get_args()
    # training hyperparameters
    batch_size = args.bs
    n_epoch = args.epochs
    l_rate = args.lr
    data_folder = Path(args.data_folder)

    # batch_size = 4  # 4 for testing, 16 for training
    # n_epoch = 10
    # l_rate = 1e-5  #
    # data_folder = r'C:\Users\sebas\Documents\MATLAB\DataProCiencia\DeepLearning'

    # Loading Data
    dataset = SpectrumDataset(data_folder/'train')
    print(f'Dataset length: {len(dataset)}')
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print(f'Dataloader length: {len(train_loader)}')

    # DDPM noise schedule
    time_steps = 1000
    betas = gd.get_named_beta_schedule('linear', time_steps)
    diffusion = gd.SpacedDiffusion(
        use_timesteps = gd.space_timesteps(time_steps, section_counts=[time_steps]),
        betas = betas,
        model_mean_type = gd.ModelMeanType.EPSILON,
        model_var_type= gd.ModelVarType.FIXED_LARGE,
        loss_type = gd.LossType.MSE,
        rescale_timesteps = True,
    )
    
    # Model and optimizer
    nn_model = UNETv2(in_channels=80, residual=False, attention_res=[]).to(device)
    print("Num params: ", sum(p.numel() for p in nn_model.parameters() if p.requires_grad))

    optim = torch.optim.Adam(nn_model.parameters(), lr=l_rate)

    trained_epochs = 0
    if trained_epochs > 0:
        nn_model.load_state_dict(torch.load(save_dir/f"model_{trained_epochs}.pth", map_location=device))  # From last model
        loss_arr = np.load(save_dir/f"loss_{trained_epochs}.npy").tolist()  # From last model
    else:
        loss_arr = []

    # Training
    nn_model.train()
    # pbar = tqdm(range(trained_epochs+1,n_epoch+1), mininterval=2)
    print(f' Epoch {trained_epochs}/{n_epoch}, {datetime.now()}')
    for ep in range(trained_epochs+1, n_epoch+1):
        # pbar = tqdm(train_loader, mininterval=2)
        for x, y in train_loader:  # x: images
            optim.zero_grad()
            x = x.to(device)
            y = y.to(device)

            # perturb data
            noise = torch.randn_like(y)
            t = torch.randint(0, time_steps, (x.shape[0],)).to(device)
            y_pert = diffusion.q_sample(y, t, noise)
            
            # use network to recover noise
            predicted_noise = nn_model(x, y_pert, t)

            # loss is mean squared error between the predicted and true noise
            loss = func.mse_loss(predicted_noise, noise)
            loss.backward()

            # nn.utils.clip_grad_norm_(nn_model.parameters(),0.5)
            loss_arr.append(loss.item())
            optim.step()

        print(f' Epoch {ep:03}/{n_epoch}, loss: {loss_arr[-1]:.2f}, {datetime.now()}')
        # save model every x epochs
        if ep % 5 == 0 or ep == int(n_epoch):
            torch.save(nn_model.state_dict(), save_dir/f"model_{ep}.pth")
            np.save(save_dir/f"loss_{ep}.npy", np.array(loss_arr))

def get_args():
    parser = argparse.ArgumentParser(description='Trains diffusion model')
    parser.add_argument('data_folder', type=str, help='parent folder where data is stored')
    parser.add_argument('bs', type=int, help='batch size')
    parser.add_argument('epochs', type=int, help='number of epochs')
    parser.add_argument('lr', type=float, help='learning rate')
    return parser.parse_args() 


if __name__ == '__main__':
    main()
