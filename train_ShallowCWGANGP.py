import sys
from shallow_CWGANGP import ShallowDiscriminator, ShallowGenerator
from tools import *
from data import TorchMelDataset, load_yaml, save_yaml
import argparse


def train_ShallowCWGANGP():
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', default='config_Sh_CWGAN_GP.yaml')
    parser.add_argument('--checkpoint', default='log/checkpoint.yaml')
    parser.add_argument('--load_from_checkpoint', default=False, action='store_true')

    a = parser.parse_args()

    config = load_yaml(a.config)
    checkpoint_config = load_yaml(a.checkpoint)

    # To store the fully trained model for inference or further training
    model = load_yaml('log/models.yaml')

    # Device to run the computations on
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

    # Load the dataset
    dataset = TorchMelDataset(config=config)

    # Get the input shape of a Mel Spectrogram
    mel, _, _, = dataset[0]
    input_shape = mel[0].shape

    # Get the number of batches
    if len(dataset) % config['batch_size'] != 0:
        num_batches = int(len(dataset) / config['batch_size']) + 1
    else:
        num_batches = int(len(dataset) / config['batch_size'])

    # Prepare the dataloader
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=config['batch_size'],
                                             shuffle=config['shuffle'],
                                             num_workers=config['num_workers'])

    classes = config['num_class']

    gen_list = []
    dis_list = []
    optimG_list = []
    optimD_list = []

    dis_loss_list = []
    gen_loss_list = []

    gen_state_dict = [{}] * config['num_strips']
    dis_state_dict = [{}] * config['num_strips']
    optim_gen_state_dict = [{}] * config['num_strips']
    optim_dis_state_dict = [{}] * config['num_strips']

    betas = config['betas']
    # If training from checkpoint, be sure to not forget the previous number of strips defined
    for idx in range(config['num_strips']):
        gen_list.append(ShallowGenerator(input_shape=input_shape,
                                         z_dim=config['latent_dim'],
                                         classes=classes).to(device))

        dis_list.append(ShallowDiscriminator(input_shape=input_shape,
                                             classes=classes,
                                             fm_idx=config['dis_fm_idx']).to(device))

        dis_loss_list.append([])
        gen_loss_list.append([])

    for idx in range(config['num_strips']):
        optimG_list.append(torch.optim.Adam(params=gen_list[idx].parameters(), lr=config['learning_rate'], betas=betas))
        optimD_list.append(torch.optim.Adam(params=dis_list[idx].parameters(), lr=config['learning_rate'], betas=betas))

    current_epoch = 0

    if a.load_from_checkpoint:

        chkpt = torch.load(checkpoint_config['checkpoint_file'])
        for idx in range(config['num_strips']):
            dis_list[idx].load_state_dict(chkpt['dis_state_dict'][idx])
            optimD_list[idx].load_state_dict(chkpt['optim_dis_state_dict'][idx])

            gen_list[idx].load_state_dict(chkpt['gen_state_dict'][idx])
            optimG_list[idx].load_state_dict(chkpt['optim_gen_state_dict'][idx])

        dis_loss_list = chkpt['dis_loss_stats']
        gen_loss_list = chkpt['gen_loss_stats']

        current_epoch = chkpt['epoch'] + 1  # Resume one epoch after previous

    else:
        # Apply the recommended weight initialization in DCGAN paper
        for idx in range(config['num_strips']):
            gen_list[idx].apply(weights_init)
            dis_list[idx].apply(weights_init)

    # torch.autograd.set_detect_anomaly(True)

    # Symbols and notations used for variable naming are based on the CVAE GANS paper
    # Including batches, the shape of the data is size([batch, n_strip, ch, h, w])
    for epoch in range(max(0, current_epoch), config['epochs']):
        # data is in batches. Output from D and C are also in batches instead of individual predictions
        for i, (data, _, labels) in enumerate(dataloader):
            batch_size = data.shape[0]

            # Prepare the class labels
            cr = []
            for idx in range(len(classes)):
                if len(classes) == 1:
                    cr.append(labels[:, idx + 1].unsqueeze(1))
                else:
                    cr.append(labels[:, idx].unsqueeze(1))
            cr = torch.stack(cr, 0).to(device)

            xr = data.to(device)

            for _ in range(config['n_critic']):

                zf = torch.randn(batch_size, config['latent_dim']).to(device)

                for idx in range(config['num_strips']):
                    dis_list[idx].zero_grad()

                    xf = gen_list[idx](zf, cr)
                    critic_real, _ = dis_list[idx](xr[:, idx], cr)
                    critic_fake, _ = dis_list[idx](xf, cr)
                    gp = gradient_penalty(dis_list[idx], xr[:, idx], xf, cr, device)
                    critic_loss = -(torch.mean(critic_real) - torch.mean(critic_fake)) + config['lambda_gp'] * gp

                    dis_loss_list[idx].append(critic_loss.item())

                    critic_loss.backward()
                    optimD_list[idx].step()

            # Training generator: min -E[discriminator(xf)]
            zf = torch.randn(batch_size, config['latent_dim']).to(device)

            for idx in range(config['num_strips']):
                gen_list[idx].zero_grad()

                xf = gen_list[idx](zf, cr)

                critic_generator, _ = dis_list[idx](xf, cr)

                gen_loss = -torch.mean(critic_generator)

                gen_loss_list[idx].append(gen_loss.item())

                gen_loss.backward()
                optimG_list[idx].step()

            # Store the state dict for every gen, dis and optim
            for idx in range(config['num_strips']):
                gen_state_dict[idx] = gen_list[idx].state_dict()
                dis_state_dict[idx] = dis_list[idx].state_dict()
                optim_dis_state_dict[idx] = optimD_list[idx].state_dict()
                optim_gen_state_dict[idx] = optimG_list[idx].state_dict()

            # Save the model at this checkpoint
            if checkpoint_config['checkpoint_within_epoch'] == 1:
                if i % checkpoint_config['checkpoint_interval_batch'] == 0 and i != 0:
                    file_name = "Shallow_CWGANGP_" + str(epoch) + "_" + str(i) + "_of_" + str(num_batches - 1) + ".pt"

                    checkpoint_config['checkpoint_file'] = checkpoint_config['checkpoint_dir'] + file_name

                    torch.save({
                        'epoch': epoch,
                        'dis_state_dict': dis_state_dict,
                        'optim_dis_state_dict': optim_dis_state_dict,
                        'gen_state_dict': gen_state_dict,
                        'optim_gen_state_dict': optim_gen_state_dict,
                        'dis_loss_stats': dis_loss_list,  # Allows post analysis of the discriminator loss
                        'gen_loss_stats': gen_loss_list,  # Allows post analysis of the generator loss
                    }, checkpoint_config['checkpoint_file'])

                    save_yaml(checkpoint_config, 'log/checkpoint.yaml')

            loss_msg = '------ '
            for idx in range(config['num_strips']):
                loss_msg = loss_msg + f"dis[{idx}]: {round(dis_loss_list[idx][-1], 4)} ------ " +\
                           f"gen[{idx}]: {round(gen_loss_list[idx][-1], 4)} ------ \n" + " ------ "

            msg = f"=======================================================================================" \
                  + f"===============================\n" \
                  + f"Epoch: {epoch} ---- Batch: {i}/{num_batches - 1}\n" \
                  + loss_msg \
                  + f"\n======================================================================================" \
                  + f"================================"

            sys.stdout.write(msg)
            sys.stdout.write('\033[' + str(config['num_strips'] + 3) + 'F')
            sys.stdout.flush()

        print("\n\n")

        if checkpoint_config['checkpoint_at_epoch'] == 1:
            if epoch % checkpoint_config['checkpoint_interval_epoch'] == 0:
                file_name = "Shallow_CWGANGP_" + str(epoch) + ".pt"

                checkpoint_config['checkpoint_file'] = checkpoint_config['checkpoint_dir'] + file_name

                torch.save({
                    'epoch': epoch,
                    'dis_state_dict': dis_state_dict,
                    'optim_dis_state_dict': optim_dis_state_dict,
                    'gen_state_dict': gen_state_dict,
                    'optim_gen_state_dict': optim_gen_state_dict,
                    'dis_loss_stats': dis_loss_list,  # Allows post analysis of the discriminator loss
                    'gen_loss_stats': gen_loss_list,  # Allows post analysis of the generator loss
                }, checkpoint_config['checkpoint_file'])

                save_yaml(checkpoint_config, 'log/checkpoint.yaml')

    # Save the model after the final epoch is reached
    model['filename'] = "Shallow_CWGANGP_" + str(config['epochs']) + ".pt"

    torch.save({
        'epoch': config['epochs'],
        'dis_state_dict': dis_state_dict,
        'optim_dis_state_dict': optim_dis_state_dict,
        'gen_state_dict': gen_state_dict,
        'optim_gen_state_dict': optim_gen_state_dict,
        'dis_loss_stats': dis_loss_list,  # Allows post analysis of the discriminator loss
        'gen_loss_stats': gen_loss_list,  # Allows post analysis of the generator loss
    }, model['dir'] + model['filename'])

    save_yaml(model, 'log/models.yaml')


if __name__ == '__main__':
    train_ShallowCWGANGP()
