import sys
from CWGANGP import Generator, Discriminator
from tools import *
from data import TorchMelDataset, load_yaml, save_yaml
import argparse


def train_CWGANGP():
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', default='config_CWGAN_GP.yaml')
    parser.add_argument('--checkpoint', default='log/checkpoint.yaml')
    parser.add_argument('--load_from_checkpoint', default=False)

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
    input_shape = mel.shape
    print(f"Mel Spectrogram shape: {input_shape}")
    print(f"Size of dataset: {len(dataset)}")

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

    # to decide the number of class for one hot encoding, config['num_class'] = [num of class A, num of class B etc.]
    classes = config['num_class']

    generator = Generator(input_shape=input_shape,
                          z_dim=config['latent_dim'],
                          classes=classes)

    discriminator = Discriminator(input_shape=input_shape,
                                  classes=classes,
                                  fm_idx=config['dis_fm_idx'])

    betas = config['betas']

    # Setup Optimizer
    optimD = torch.optim.Adam(params=discriminator.parameters(), lr=config['learning_rate'], betas=betas)
    optimG = torch.optim.Adam(params=generator.parameters(), lr=config['learning_rate'], betas=betas)

    prev_epoch = 0
    dis_loss_list = []
    gen_loss_list = []

    # Send the models to a desired device
    generator.to(device)
    discriminator.to(device)

    if a.load_from_checkpoint:

        chkpt = torch.load(checkpoint_config['checkpoint_file'])

        discriminator.load_state_dict(chkpt['dis_state_dict'])
        optimD.load_state_dict(chkpt['optim_dis_state_dict'])
        dis_loss_list = chkpt['dis_loss_stats']

        generator.load_state_dict(chkpt['gen_state_dict'])
        optimG.load_state_dict(chkpt['optim_gen_state_dict'])
        gen_loss_list = chkpt['gen_loss_stats']

        prev_epoch = chkpt['epoch'] + 1  # Resume at epoch + 1 (current epoch already completed)

    else:
        # Apply the recommended weight initialization in DCGAN paper
        generator.apply(weights_init)
        discriminator.apply(weights_init)

    # torch.autograd.set_detect_anomaly(True)

    # Symbols and notations used for variable naming are based on the CVAE GANS paper
    for epoch in range(max(0, prev_epoch), config['epochs']):
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

            critic_loss = None
            for _ in range(config['n_critic']):
                discriminator.zero_grad()

                zf = torch.randn(batch_size, config['latent_dim']).to(device)

                xf = generator(zf, cr)

                critic_real, _ = discriminator(xr, cr)
                critic_fake, _ = discriminator(xf, cr)
                gp = gradient_penalty(discriminator, xr, xf, cr, device)
                critic_loss = -(torch.mean(critic_real) - torch.mean(critic_fake)) + config['lambda_gp'] * gp

                critic_loss.backward()
                optimD.step()

            # Training generator: min -E[discriminator(xf)]
            generator.zero_grad()

            zf = torch.randn(batch_size, config['latent_dim']).to(device)
            xf = generator(zf, cr)

            critic_generator, fd_xf = discriminator(xf, cr)  # generator validity score not in use

            gen_loss = -torch.mean(critic_generator)
            gen_loss.backward()
            optimG.step()

            dis_loss_val = critic_loss.item()
            dis_loss_list.append(dis_loss_val)
            gen_loss_val = gen_loss.item()
            gen_loss_list.append(gen_loss_val)

            # Save the model at this checkpoint
            if checkpoint_config['checkpoint_within_epoch'] == 1:
                if i % checkpoint_config['checkpoint_interval_batch'] == 0 and i != 0:
                    file_name = "CWGANGP_" + str(epoch) + "_" + str(i) + "_of_" + str(num_batches - 1) + ".pt"

                    checkpoint_config['checkpoint_file'] = checkpoint_config['checkpoint_dir'] + file_name

                    torch.save({
                        'epoch': epoch,
                        'dis_state_dict': discriminator.state_dict(),
                        'optim_dis_state_dict': optimD.state_dict(),
                        'gen_state_dict': generator.state_dict(),
                        'optim_gen_state_dict': optimG.state_dict(),
                        'dis_loss_stats': dis_loss_list,  # Allows post analysis of the discriminator loss
                        'gen_loss_stats': gen_loss_list,  # Allows post analysis of the generator loss
                    }, checkpoint_config['checkpoint_file'])

                    save_yaml(checkpoint_config, 'log/checkpoint.yaml')

            msg = f"=======================================================================================" \
                  + f"===============================\n" \
                  + f"Epoch: {epoch} ---- Batch: {i}/{num_batches - 1}\n" \
                  + f"dis loss: {round(dis_loss_val, 4)}" \
                  + f" ---- gen loss: {round(gen_loss.item(), 4)}" \
                  + f"======================================================================================" \
                  + f"================================"

            sys.stdout.write(msg)
            sys.stdout.write('\033[3F')
            sys.stdout.flush()

        print("\n\n")

        if checkpoint_config['checkpoint_at_epoch'] == 1:
            if epoch % checkpoint_config['checkpoint_interval_epoch'] == 0 and epoch != 0:
                file_name = "CWGANGP_" + str(epoch) + ".pt"

                checkpoint_config['checkpoint_file'] = checkpoint_config['checkpoint_dir'] + file_name

                torch.save({
                    'epoch': epoch,
                    'dis_state_dict': discriminator.state_dict(),
                    'optim_dis_state_dict': optimD.state_dict(),
                    'gen_state_dict': generator.state_dict(),
                    'optim_gen_state_dict': optimG.state_dict(),
                    'dis_loss_stats': dis_loss_list,  # Allows post analysis of the discriminator loss
                    'gen_loss_stats': gen_loss_list,  # Allows post analysis of the generator loss
                }, checkpoint_config['checkpoint_file'])

                save_yaml(checkpoint_config, 'log/checkpoint.yaml')

    # Save the model after the final epoch is reached
    model['filename'] = "CWGANGP_" + str(config['epochs']) + ".pt"

    torch.save({
        'epoch': config['epochs'],
        'dis_state_dict': discriminator.state_dict(),
        'optim_dis_state_dict': optimD.state_dict(),
        'gen_state_dict': generator.state_dict(),
        'optim_gen_state_dict': optimG.state_dict(),
        'dis_loss_stats': dis_loss_list,  # Allows post analysis of the discriminator loss
        'gen_loss_stats': gen_loss_list,  # Allows post analysis of the decoder loss
    }, model['dir'] + model['filename'])

    save_yaml(model, 'log/models.yaml')


if __name__ == '__main__':
    train_CWGANGP()
