from data import *
import matplotlib.pyplot as plt
import matplotlib
import argparse
matplotlib.use('Qt5Agg')  # Use this backend to support interactive display

parser = argparse.ArgumentParser()

parser.add_argument('--config', default='config_CWGAN_GP.yaml')
parser.add_argument('--checkpoint_model', default='log/models/CWGANGP_D/CWGANGP1000.pt')

a = parser.parse_args()


# A template to use the monitor the training plots of the checkpoint models
def training_plot():
    config = load_yaml(a.config)
    model_info_dict = torch.load(a.checkpoint_model)

    gen_loss = model_info_dict['gen_loss_stats']
    dis_loss = model_info_dict['dis_loss_stats']

    plt.figure(0)
    plt.plot(range(len(gen_loss)), gen_loss)
    plt.plot(range(len(dis_loss)), dis_loss)
    # plt.ylim([-40, 100])
    plt.xlim([0, len(gen_loss) - 1])
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend(["Gen Loss", "Critic Loss"], loc='upper right', fontsize='x-small')

    plt.show()


if __name__ == '__main__':
    training_plot()
