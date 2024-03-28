import matplotlib.pyplot as plt
from IPython.display import HTML
from torchvision import transforms
from matplotlib.animation import FuncAnimation

def show_tensor_image(image, vmin=0.2, vmax=1.6):
    reverse_transforms = transforms.Compose([
        # transforms.Lambda(lambda t: (t + 1)),
        transforms.Lambda(lambda t: (t*0.17 + 0.95)),
        transforms.Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC
        # transforms.Lambda(lambda t: (t * 60) - 60.),
        transforms.Lambda(lambda t: t.numpy())
    ])
    # Take first image of batch
    if len(image.shape) == 4:
        image = image[0, :, :, :]
    extent_ACS = (-1.755, 1.785, 3.7614375, 0.1783125)
    # extent_ACS = (-1.845, 1.845, 3.751, 0.128)
    plt.imshow(reverse_transforms(image), cmap='turbo', extent=extent_ACS)
    plt.clim(vmin,vmax)

def show_reverse_process(intermediate):
    """ Shows a list of tensors from the sampling process """
    num_intermediate = len(intermediate)
    plt.figure(figsize=(16,2))
    plt.axis('off')
    for id, y_gen in enumerate(intermediate):
        plt.subplot(1, num_intermediate, id+1)
        show_tensor_image(y_gen)
    plt.show()


# def animate_reverse_process(intermediate: list):
#     """ Animates a list of tensors from the sampling process. Asumes BATCH_SIZE = 4 """
#     fig, axs = plt.subplots(2, 2, figsize=(6, 6))
#     for iSample in range(4):
#         current_ax = axs[iSample//2][iSample%2]
#         current_ax.set_title(f'Sample {iSample}')
#         artist_im = current_ax.imshow(intermediate[0,iSample,0,:,:], extent=[-20,20,50,0], cmap='gray')
#         artist_im.set_clim(-1,1)
#         plt.colorbar(mappable=artist_im,ax=current_ax)
#     plt.tight_layout()

#     def update(frame):
#         artist_arr = []
#         for iSample in range(4):
#             current_ax = axs[iSample//2][iSample%2]
#             artist_im = current_ax.imshow(intermediate[frame,iSample,0,:,:], extent=[-20,20,50,0], cmap='gray')
#             artist_im.set_clim(0,1)
#             artist_arr.append(artist_im)
#         return artist_arr

#     num_frames = intermediate.shape[0]
#     print("Creating animation...")
#     ani = FuncAnimation(fig=fig, func=update, frames=num_frames, interval=30)
#     plt.close()
#     return HTML(ani.to_jshtml())


# def plot_minibatch(samples, title="Sample"):
#     fig, axs = plt.subplots(2, 2, figsize=(6, 6))
#     for iSample in range(4):
#         im = axs[iSample//2][iSample%2].imshow(samples[iSample,0,:,:].to('cpu'), extent=[-20,20,50,0],cmap='gray')
#         im.set_clim(-1,1)
#         axs[iSample//2][iSample%2].set_title(f'{title} {iSample}')
#         plt.colorbar(mappable=im,ax=axs[iSample//2][iSample%2])
#     plt.tight_layout()
#     plt.show()