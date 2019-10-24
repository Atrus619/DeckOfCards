# Various functions to check on progress of training

import util.db as db
import matplotlib.pyplot as plt
import util.util as uu
import numpy as np


def get_diagnostic_plots(run_id):
    df = db.get_metrics(run_id=run_id)
    config = uu.get_config(path=run_id)

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

    st = f.suptitle("Training Diagnostic Plots", fontsize='x-large')
    f.tight_layout()
    st.set_y(0.96)
    f.subplots_adjust(top=0.9)
