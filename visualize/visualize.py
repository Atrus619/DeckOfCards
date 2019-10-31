# Various functions to check on progress of training

import util.db as db
import matplotlib.pyplot as plt
import numpy as np
import util.util as util


def plot_diagnostic_plots(run_id):
    df = db.get_metrics(run_id=run_id)
    config = util.get_config(path=run_id)

    num_cycles = len(df)
    total_length = num_cycles * config.benchmark_freq
    x_axis = np.linspace(start=config.benchmark_freq, stop=total_length, num=num_cycles)

    f, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

    axes[0].title.set_text('Win Rate vs. Model 2')
    axes[0].plot(x_axis, df.win_rate)
    axes[0].set_ylabel('Win Rate (%)')
    axes[0].plot(np.linspace(config.benchmark_freq, total_length, total_length), np.full(total_length, 0.5), linestyle='dashed', color='r')
    for i in range(config.benchmark_freq, total_length):
        if i % config.player_2_update_freq == 0:
            axes[0].axvline(x=i, linestyle='dashed', color='g')

    axes[1].title.set_text('Win Rate vs. RandomBot')
    axes[1].plot(x_axis, df.win_rate_random)
    axes[1].set_ylabel('Win Rate (%)')
    axes[1].plot(np.linspace(config.benchmark_freq, total_length, total_length), np.full(total_length, 0.5), linestyle='dashed', color='r')

    axes[2].title.set_text('Avg Reward')
    axes[2].plot(x_axis, df.average_reward)
    axes[2].set_xlabel('Cycle')
    axes[2].set_ylabel('Reward')
    axes[2].plot(np.linspace(config.benchmark_freq, total_length, total_length), np.full(total_length, 0.0), linestyle='dashed', color='r')

    st = f.suptitle('Training Diagnostic Plots', fontsize='x-large')
    f.tight_layout()
    st.set_y(0.96)
    f.subplots_adjust(top=0.9)


def plot_model_training_plots(run_id):
    model = util.get_model_checkpoint(run_id=run_id)
    f, axes = plt.subplots(2, 1, figsize=(12, 12), sharex=True)

    axes[0].title.set_text('Avg Loss by Epoch')
    axes[0].plot(model.policy_net.losses)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')

    axes[1].title.set_text('Estimated Q by Epoch')
    axes[1].plot(model.policy_net.Qs)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Q')

    st = f.suptitle('Model Training Diagnostic Plots', fontsize='x-large')
    f.tight_layout()
    st.set_y(0.96)
    f.subplots_adjust(top=0.9)


def plot_model_layer_scatters(run_id, figsize=(20, 10)):
    model = util.get_model_checkpoint(run_id=run_id)
    net = model.policy_net

    assert net.epoch > 0, "Model needs to be trained first"

    f, axes = plt.subplots(len(net.layer_list), 4, figsize=figsize, sharex=True)

    axes[0, 0].title.set_text("Weight Norms")
    axes[0, 1].title.set_text("Weight Gradient Norms")
    axes[0, 2].title.set_text("Bias Norms")
    axes[0, 3].title.set_text("Bias Gradient Norms")

    for i in range(4):
        axes[len(net.layer_list) - 1, i].set_xlabel('epochs')

    for i, layer in enumerate(net.layer_list):
        axes[i, 0].set_ylabel(net.layer_list_names[i])
        axes[i, 0].plot(net.wnorm_history[layer]['weight'])
        axes[i, 1].plot(net.gnorm_history[layer]['weight'])
        axes[i, 2].plot(net.wnorm_history[layer]['bias'])
        axes[i, 3].plot(net.gnorm_history[layer]['bias'])

    sup = net.name + " Layer Weight and Gradient Norms"
    st = f.suptitle(sup, fontsize='x-large')
    f.tight_layout()
    st.set_y(0.96)
    f.subplots_adjust(top=0.9)


def plot_model_layer_hists(run_id, epoch=None, figsize=(20, 10)):
    model = util.get_model_checkpoint(run_id=run_id)
    net = model.policy_net

    if epoch is None:
        epoch = net.epoch

    f, axes = plt.subplots(len(net.layer_list), 4, figsize=figsize, sharex=False)

    axes[0, 0].title.set_text("Weight Histograms")
    axes[0, 1].title.set_text("Weight Gradient Histograms")
    axes[0, 2].title.set_text("Bias Histograms")
    axes[0, 3].title.set_text("Bias Gradient Histograms")

    for i in range(4):
        axes[len(net.layer_list) - 1, i].set_xlabel('Value')

    for i, layer in enumerate(net.layer_list):
        axes[i, 0].set_ylabel(net.layer_list_names[i])

        plt.sca(axes[i, 0])
        util.convert_np_hist_to_plot(net.histogram_weight_history[layer]['weight'][epoch])

        plt.sca(axes[i, 2])
        util.convert_np_hist_to_plot(net.histogram_weight_history[layer]['bias'][epoch])
        if epoch == 0:
            pass
        else:
            plt.sca(axes[i, 1])
            util.convert_np_hist_to_plot(net.histogram_gradient_history[layer]['weight'][epoch])

            plt.sca(axes[i, 3])
            util.convert_np_hist_to_plot(net.histogram_gradient_history[layer]['bias'][epoch])

    sup = net.name + " Layer Weight and Gradient Histograms - Epoch " + str(epoch)
    st = f.suptitle(sup, fontsize='x-large')
    f.tight_layout()
    st.set_y(0.96)
    f.subplots_adjust(top=0.9)


def compare_final_random_bot_winrates(experiment_file, avg_latest_n=3):
    experiments = util.get_experiment_file(experiment_file)
    winrates = []
    run_ids = []
    for index, experiment in experiments.iterrows():
        winrates.append(db.get_metrics(run_id=experiment.run_id).win_rate_random.iloc[-1 - avg_latest_n:-1].mean())
        run_ids.append(experiment.run_id)

    plt.bar(height=winrates, x=run_ids)

    plt.plot(np.linspace(-0.5, len(run_ids) - 0.5, len(run_ids)), np.full(len(run_ids), 0.5), linestyle='dashed', color='r')

    plt.xlabel('Run ID', fontweight='bold')
    plt.ylabel('Winrate vs. RandomBot (%)', fontweight='bold')
    plt.title(f'Avg Latest {avg_latest_n} Benchmarked Winrates Vs. RandomBot by Run ID', fontweight='bold')
