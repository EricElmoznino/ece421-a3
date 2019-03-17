import matplotlib.pyplot as plt


def line_plot(x, ys, labels, x_label, y_label, save_path):
    plt.close()
    plots = [plt.plot(x, y, label=label)[0] for y, label in zip(ys, labels)]
    plt.legend(handles=plots)
    save_plot(save_path, x_label, y_label)


def k_means_plot_2d(x, assignments, k, save_path):
    plt.close()
    x_assigned = [x[assignments == i] for i in range(k)]
    plots = [plt.scatter(x[:, 0], x[:, 1], label='k = %d' % i) for i, x in enumerate(x_assigned)]
    plt.legend(handles=plots)
    save_plot(save_path)


def data_plot_2d(x, save_path):
    plt.close()
    plot = plt.scatter(x[:, 0], x[:, 1])
    save_plot(save_path)


def save_plot(save_path, x_label=None, y_label=None):
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.savefig(save_path, bbox_inches='tight')
