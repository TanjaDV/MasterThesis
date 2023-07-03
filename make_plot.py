# import packages
import matplotlib.pyplot as plt


def plot_history(axes, history, run, runs, title):
    """
    This function makes a plot of the loss and accuracy
    The function does not return anything, but it does add information to the axes
    """
    if runs == 1:
        axes[0].plot(history.history['loss'], label="training")
        axes[0].plot(history.history['val_loss'], label="validation")
        y_max = max(1, max(history.history['loss']), max(history.history['val_loss'])) * 1.02
        axes[0].set_ylim(0, y_max)

        axes[1].plot(history.history['acc'], label="training")
        axes[1].plot(history.history['val_acc'], label="validation")
        axes[1].set_ylim(0, 1.02)
    else:
        axes[run][0].plot(history.history['loss'], label="training")
        axes[run][0].plot(history.history['val_loss'], label="validation")
        y_max = max(1, max(history.history['loss']), max(history.history['val_loss'])) * 1.02
        axes[run][0].set_ylim(0, y_max)

        axes[run][1].plot(history.history['acc'], label="training")
        axes[run][1].plot(history.history['val_acc'], label="validation")
        axes[run][1].set_ylim(0, 1.02)

        axes[run][0].annotate(title, xy=(0, 0.5), xytext=(-axes[run][0].yaxis.labelpad - 15, 0),
                              xycoords=axes[run][0].yaxis.label, textcoords='offset points',
                              size='medium', ha='right', va='center')
    return


def plot_history_finalization(dir_plot, runs, fig, axes, x_label="Epochs"):
    """
    This function finalizes the layout of the plots and saves the end result
    The function does not return anything, but it does save the final plot to a file
    """
    if runs == 1:
        axes[0].set_ylabel("SCCE loss")
        axes[1].set_ylabel("Accuracy")
        axes[0].set_xlabel(x_label)
        axes[1].set_xlabel(x_label)

        handles, labels = axes[0].get_legend_handles_labels()
        axes[1].legend(handles, labels, loc='lower right')
    else:
        axes[0][0].set_title("SCCE loss")
        axes[0][1].set_title("Accuracy")
        axes[runs - 1][0].set_xlabel(x_label)
        axes[runs - 1][1].set_xlabel(x_label)

        handles, labels = axes[1][0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower left')

    small_size = 12
    medium_size = 14
    big_size = 16

    plt.rc('font', size=small_size)  # controls default text sizes
    plt.rc('axes', labelsize=big_size)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=small_size)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=small_size)  # fontsize of the tick labels
    plt.rc('legend', fontsize=medium_size)  # legend fontsize

    fig.tight_layout()
    fig.savefig(dir_plot + "history_plot")
    plt.clf()
    plt.cla()
    return


def plot_comparison(dir_plot, iterable, loss1, loss2, acc1, acc2, label1, label2, title, x_label,
                    x_ticks=None, y_ticks=None):
    plot_comparison_acc(dir_plot, iterable, acc1, acc2, label1, label2, title, x_label,
                        x_ticks, y_ticks)
    plot_comparison_all(dir_plot, iterable, loss1, loss2, acc1, acc2, label1, label2, title, x_label)

    plot_comparison_zoomed_acc(dir_plot, iterable, acc1, acc2, label1, label2, title, x_label,
                               x_ticks, y_ticks)
    return


def plot_comparison_all(dir_plot, iterable, loss1, loss2, acc1, acc2, label1, label2, title, x_label):
    fig, axes = plt.subplots(nrows=1, ncols=2, sharex='col')

    axes[0].plot(iterable, loss1, label=label1)
    axes[0].plot(iterable, loss2, label=label2, linestyle='dashed')
    y_max = max(1, max(loss1), max(loss2)) * 1.02
    axes[0].set_ylim(0, y_max)

    axes[1].plot(iterable, acc1, label=label1)
    axes[1].plot(iterable, acc2, label=label2)
    axes[1].set_ylim(0, 1.02)

    plot_history_finalization(dir_plot + title, 1, fig, axes, x_label)
    return


def plot_comparison_acc(dir_plot, iterable, acc1, acc2, label1, label2, title, x_label, x_ticks, y_ticks):
    plt.plot(iterable, acc1, label=label1, linewidth=2)
    plt.plot(iterable, acc2, label=label2, linewidth=2, linestyle='dashdot')
    plt.ylim(0, 1.02)
    plt.ylabel("Accuracy")
    plt.xlabel(x_label)

    if x_ticks:
        plt.gca().xaxis.set_major_locator(plt.MultipleLocator(x_ticks))
    if y_ticks:
        plt.gca().yaxis.set_major_locator(plt.MultipleLocator(y_ticks))
    plt.legend(loc='lower right')

    plt.tight_layout()
    plt.savefig(dir_plot + title + "history_acc_plot")
    plt.close()
    plt.clf()
    return


def plot_comparison_zoomed_acc(dir_plot, iterable, acc1, acc2, label1, label2, title, x_label, x_ticks, y_ticks):
    if title != "effectpretext_":
        return

    plt.plot(iterable, acc1, label=label1, linewidth=2)
    plt.plot(iterable, acc2, label=label2, linewidth=2, linestyle='dashdot')

    if label1 == "downstream":
        y_min = min(acc1) - 0.01
        y_max = max(acc1) + 0.01
    else:
        y_min = min(acc2) - 0.01
        y_max = max(acc2) + 0.01

    plt.ylim(y_min, y_max)
    plt.ylabel("Accuracy")
    plt.xlabel(x_label)

    if x_ticks:
        plt.gca().xaxis.set_major_locator(plt.MultipleLocator(x_ticks))
    if y_ticks:
        plt.gca().yaxis.set_major_locator(plt.MultipleLocator(y_ticks / 10))
    plt.legend(loc='lower right')

    plt.tight_layout()
    plt.savefig(dir_plot + title + "history_zoomed_acc_plot")
    plt.close()
    plt.clf()
    return


def plot_comparison_three(dir_plot, iterable, acc1, acc2, acc3, label1, label2, label3,
                          title, x_label, x_ticks, y_ticks):
    plt.plot(iterable, acc1, label=label1, linewidth=2)
    plt.plot(iterable, acc2, label=label2, linewidth=2, linestyle='dashdot')
    plt.plot(iterable, acc3, label=label3, linewidth=2, linestyle='dotted')
    plt.ylim(0, 1.02)
    plt.ylabel("Accuracy")
    plt.xlabel(x_label)

    if x_ticks:
        plt.gca().xaxis.set_major_locator(plt.MultipleLocator(x_ticks))
    if y_ticks:
        plt.gca().yaxis.set_major_locator(plt.MultipleLocator(y_ticks))
    plt.legend(loc='lower right')

    plt.tight_layout()
    plt.savefig(dir_plot + title + "history_acc_plot")
    plt.close()
    plt.clf()
    return
