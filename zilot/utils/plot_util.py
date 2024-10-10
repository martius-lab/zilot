from tempfile import SpooledTemporaryFile

import matplotlib.pyplot as plt
import numpy as np
import torch
from imageio import imread
from matplotlib.legend_handler import HandlerBase
from matplotlib.patches import Rectangle


class HandlerColormap(HandlerBase):
    def __init__(self, cmap, num_stripes=8, **kw):
        HandlerBase.__init__(self, **kw)
        self.cmap = cmap
        self.num_stripes = num_stripes

    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        stripes = []
        for i in range(self.num_stripes):
            s = Rectangle(
                [xdescent + i * width / self.num_stripes, ydescent],
                width / self.num_stripes,
                height,
                fc=self.cmap((2 * i + 1) / (2 * self.num_stripes)),
                transform=trans,
            )
            stripes.append(s)
        return stripes


def add_colorbars_to_legend(ax: plt.Axes, cmaps: list, labels: list, num_stripes: int = 16, **legend_kwargs):
    ax.legend()
    if ax.legend_ is None:
        return
    # add a patch with the colormap to the legend
    handles = ax.legend_.legend_handles
    all_labels = [h.get_label() for h in handles]
    all_labels.extend(labels)
    handler_map = {}
    for cmap in cmaps:
        handles.append(plt.Rectangle((0, 0), 1, 1))
        handler_map[handles[-1]] = HandlerColormap(cmap, num_stripes=num_stripes)
    ax.legend(handles=handles, labels=all_labels, handler_map=handler_map, **legend_kwargs)


MAX_SPOOLED_FILE_SIZE = 1024**2  # 1 MB


def fig2data(fig: plt.Figure, close=True) -> np.ndarray:
    """
    Convert a Matplotlib figure to a 3D numpy array with RGBA channels and closes the figure.
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    with SpooledTemporaryFile(max_size=MAX_SPOOLED_FILE_SIZE) as fp:
        fig.savefig(fp, format="png")
        if close:
            plt.close(fig)
        fp.seek(0)
        img = imread(fp, format="png")
    return img


def plot_value_function(
    states, goal, values, cmap="plasma", ax=None, x_coord=0, y_coord=1, vmin=None, vmax=None, **scatter_kwargs
) -> plt.Axes:
    if ax is None:
        ax = plt.gca()

    x = states[:, x_coord]
    y = states[:, y_coord]
    z = values
    vmin = vmin if vmin is not None else z.min()
    vmax = vmax if vmax is not None else z.max()
    # scatter plot of x and y with z as color and with colorbar (using seaborn)
    cmap = plt.cm.get_cmap(cmap)
    plt.colorbar(
        ax.scatter(x, y, c=z, cmap=cmap, label="States", vmin=vmin, vmax=vmax, **scatter_kwargs),
        ax=ax,
        label="Value Function",
    )
    ax.scatter(goal[x_coord], goal[y_coord], color="red", label="Goal", marker="x", s=100, linewidth=3)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    return ax


def plot_policy(states, goal, actions, ax=None, dt=1, **quiver_kwargs) -> plt.Axes:
    if ax is None:
        ax = plt.gca()
    x = states[:, 0]
    y = states[:, 1]
    u = actions[:, 0] * dt
    v = actions[:, 1] * dt
    ax.quiver(x, y, u, v, units="xy", scale=1, color="blue", label="Policy", **quiver_kwargs)
    ax.scatter(goal[0], goal[1], color="red", label="Goal", marker="x", s=100, linewidth=3)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    return ax


if __name__ == "__main__":
    states = torch.rand(100, 2)
    goal = torch.rand(2)
    value_function = torch.rand(100) * 10
    actions = torch.rand(100, 2) * 2 - 1

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 20))
    plot_value_function(states, goal, value_function, ax=ax1)
    plot_policy(states, goal, actions, ax=ax2, dt=0.1)
    plt.show()
