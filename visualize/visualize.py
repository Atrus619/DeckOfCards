# Various functions to check on progress of training

import util.db as db
import matplotlib.pyplot as plt
import numpy as np
import util.util as util
from classes.Epsilon import Epsilon


def plot_diagnostic_plots(run_id, alpha=0.5):
    df = db.get_metrics(run_id=run_id)
    config = util.get_config(path=run_id)

    num_cycles = len(df)
    total_length = num_cycles * config.benchmark_freq
    x_axis = np.linspace(start=config.benchmark_freq, stop=total_length, num=num_cycles)
    x_line_coord = np.linspace(config.benchmark_freq, total_length, total_length)

    epsilon = Epsilon(epsilon_func=config.epsilon_func, max_epsilon=config.max_epsilon, min_epsilon=config.min_epsilon,
                      eval_epsilon=config.eval_epsilon, num_cycles=config.num_cycles, decrement=config.epsilon_decrement)
    epsilon_func = epsilon.get_epsilon
    vfunc = np.vectorize(epsilon_func)
    epsilon_plot_ys = vfunc(x_line_coord)

    f, axes = plt.subplots(5, 1, figsize=(12, 12), sharex=True)

    axes[0].title.set_text('Win Rate vs. Model 2')
    axes[0].plot(x_axis, df.win_rate)
    axes[0].set_ylabel('Win Rate (%)')
    axes[0].plot(x_line_coord, np.full(total_length, 0.5), linestyle='dashed', color='r', alpha=alpha)
    axes[0].plot(x_line_coord, np.full(total_length, 1.0), linestyle='dashed', color='g', alpha=alpha)
    for i in range(config.benchmark_freq, total_length):
        if i % config.player_2_update_freq == 0:
            axes[0].axvline(x=i, linestyle='dashed', color='g', alpha=alpha)

    axes[1].title.set_text('Win Rate vs. RandomBot')
    axes[1].plot(x_axis, df.win_rate_random)
    axes[1].set_ylabel('Win Rate (%)')
    axes[1].plot(x_line_coord, np.full(total_length, 0.5), linestyle='dashed', color='r', alpha=alpha)
    axes[1].plot(x_line_coord, np.full(total_length, 1.0), linestyle='dashed', color='g', alpha=alpha)

    axes[2].title.set_text('Win Rate vs. Expert Policy')
    axes[2].plot(x_axis, df.win_rate_expert_policy)
    axes[2].set_ylabel('Win Rate (%)')
    axes[2].plot(x_line_coord, np.full(total_length, 0.5), linestyle='dashed', color='r', alpha=alpha)
    axes[2].plot(x_line_coord, np.full(total_length, 1.0), linestyle='dashed', color='g', alpha=alpha)

    axes[3].title.set_text('Avg Reward')
    axes[3].plot(x_axis, df.average_reward)
    axes[3].set_ylabel('Reward')
    axes[3].plot(x_line_coord, np.full(total_length, 0.0), linestyle='dashed', color='r', alpha=alpha)

    axes[4].title.set_text('Epsilon Schedule')
    axes[4].plot(x_line_coord, epsilon_plot_ys)
    axes[4].plot(x_line_coord, np.full(total_length, 1.0), linestyle='dashed', color='g', alpha=alpha)
    axes[4].plot(x_line_coord, np.full(total_length, 0.0), linestyle='dashed', color='r', alpha=alpha)
    axes[4].set_ylabel('Epsilon')
    axes[4].set_xlabel('Cycle')

    st = f.suptitle('Training Diagnostic Plots', fontsize='x-large')
    f.tight_layout()
    st.set_y(0.96)
    f.subplots_adjust(top=0.9)


def plot_model_training_plots(run_id, alpha=0.0):
    model = util.get_model_checkpoint(run_id=run_id)
    config = util.get_config(path=run_id)

    f, axes = plt.subplots(2, 1, figsize=(12, 12), sharex=True)

    axes[0].title.set_text('Avg Loss by Epoch')
    axes[0].plot(model.policy_net.losses)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')

    for i in range(config.epochs_per_cycle, model.policy_net.epoch, config.epochs_per_cycle):
        axes[0].axvline(x=i, linestyle='dashed', color='g', alpha=alpha)

    axes[1].title.set_text('Estimated Q by Epoch')
    axes[1].plot(model.policy_net.Qs)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Q')

    for i in range(config.epochs_per_cycle, model.policy_net.epoch, config.epochs_per_cycle):
        axes[1].axvline(x=i, linestyle='dashed', color='g', alpha=alpha)

    st = f.suptitle('Model Training Diagnostic Plots', fontsize='x-large')
    f.tight_layout()
    st.set_y(0.96)
    f.subplots_adjust(top=0.9)


def plot_model_layer_scatters(run_id, figsize=(20, 10), alpha=0.0):
    model = util.get_model_checkpoint(run_id=run_id)
    config = util.get_config(path=run_id)
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

    for i, layer in enumerate(net.layer_list):
        for j in range(4):
            for k in range(config.epochs_per_cycle, model.policy_net.epoch, config.epochs_per_cycle):
                axes[i, j].axvline(x=k, linestyle='dashed', color='g', alpha=alpha)

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


def compare_final_benchmark_winrates(experiment_file, avg_latest_n=3, alpha=0.5):
    experiments = util.get_experiment_file(experiment_file)
    random_bot_winrates = []
    expert_policy_bot_winrates = []
    run_ids = []
    for index, experiment in experiments.iterrows():
        current_metrics = db.get_metrics(run_id=experiment.run_id)
        random_bot_winrates.append(current_metrics.win_rate_random.iloc[-1 - avg_latest_n:-1].mean())
        expert_policy_bot_winrates.append(current_metrics.win_rate_expert_policy.iloc[-1 - avg_latest_n:-1].mean())
        run_ids.append(experiment.run_id)

    f, axes = plt.subplots(2, 1, figsize=(12, 12), sharex=True)

    axes[0].bar(height=random_bot_winrates, x=run_ids)
    axes[0].plot(np.linspace(-0.5, len(run_ids) - 0.5, len(run_ids)), np.full(len(run_ids), 0.5), linestyle='dashed', color='r', alpha=alpha)
    axes[0].plot(np.linspace(-0.5, len(run_ids) - 0.5, len(run_ids)), np.full(len(run_ids), 1.0), linestyle='dashed', color='g', alpha=alpha)
    axes[0].title.set_text('RandomBot')
    axes[0].set_ylabel('Winrate vs. RandomBot (%)')

    axes[1].bar(height=expert_policy_bot_winrates, x=run_ids)
    axes[1].plot(np.linspace(-0.5, len(run_ids) - 0.5, len(run_ids)), np.full(len(run_ids), 0.5), linestyle='dashed', color='r', alpha=alpha)
    axes[1].plot(np.linspace(-0.5, len(run_ids) - 0.5, len(run_ids)), np.full(len(run_ids), 1.0), linestyle='dashed', color='g', alpha=alpha)
    axes[1].title.set_text('ExpertPolicy')
    axes[1].set_ylabel('Winrate vs. ExpertPolicy (%)')
    axes[1].set_xlabel('Run ID')

    st = f.suptitle(f'Avg Latest {avg_latest_n} Benchmarked Winrates by Run ID', fontsize='x-large')
    f.tight_layout()
    st.set_y(0.96)
    f.subplots_adjust(top=0.9)
