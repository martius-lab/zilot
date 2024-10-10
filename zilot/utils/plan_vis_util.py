import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib import cm
from matplotlib.collections import LineCollection
from tqdm.auto import tqdm

from zilot.utils.plot_util import fig2data


def render_traj(traj: list[list[dict[str, np.ndarray]]]):
    x_min = min([np.min(t["x"][..., 0]) for tt in traj for t in tt])
    x_max = max([np.max(t["x"][..., 0]) for tt in traj for t in tt])
    y_min = min([np.min(t["x"][..., 1]) for tt in traj for t in tt])
    y_max = max([np.max(t["x"][..., 1]) for tt in traj for t in tt])

    v_min = min([np.min(t["value"]) for tt in traj for t in tt])
    v_max = max([np.max(t["value"]) for tt in traj for t in tt])
    norm = plt.Normalize(v_min, v_max)

    def draw_frame(tt):
        fig, axs = plt.subplots(1, len(tt), figsize=(len(tt) * 5, 5))
        for t, ax in zip(tt, axs):
            x, v = t["x"], t["value"]
            lc = LineCollection(list(x), cmap=cm.viridis, norm=norm)
            lc.set_array(v)
            ax.add_collection(lc)
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            _ = plt.colorbar(lc, ax=ax)
        return fig2data(fig)

    return np.stack([draw_frame(t) for t in tqdm(traj, desc="Drawing trajectory")])


def _weighted_bipartite_graph(weights: np.ndarray):
    H, T = weights.shape
    G = nx.Graph()
    G.add_nodes_from(range(H), bipartite=0)
    G.add_nodes_from(range(H, H + T), bipartite=1)

    for i in range(H):
        for j in range(T):
            G.add_edge(i, H + j, weight=weights[i, j])
    return G


def _draw_weighted_graph(
    ax,
    G,
    pos,
    node_color="black",
    edge_cmap=None,
    edge_cbar_label="",
    node_size=10,
    **kwargs,
):
    if isinstance(edge_cmap, str):
        edge_cmap = plt.colormaps[edge_cmap]

    if edge_cmap:
        edge_weights = [G[u][v]["weight"] for u, v in G.edges()]
        min_ew, max_ew = min(edge_weights), max(edge_weights)
        scale = max(max_ew - min_ew, 1e-12)
        alpha = [(w - min_ew) / scale for w in edge_weights]
    else:
        edge_weights = "black"
        alpha = 1.0

    nx.draw_networkx_edges(
        G,
        pos,
        ax=ax,
        width=1.0,
        alpha=alpha,
        edge_color=edge_weights,
        edge_cmap=edge_cmap,
        node_size=node_size,
        **kwargs,
    )

    nx.draw_networkx_nodes(
        G,
        pos,
        ax=ax,
        node_size=node_size,
        node_color=node_color,
    )

    if edge_cmap:
        sm = plt.cm.ScalarMappable(cmap=edge_cmap, norm=plt.Normalize(vmin=min(edge_weights), vmax=max(edge_weights)))
        sm.set_array([])
        plt.colorbar(sm, ax=ax, label=edge_cbar_label)


def _draw_line_graph(ax, points, node_color="black", edge_color="black", node_size=10, alpha=1.0):
    G = nx.DiGraph()
    G.add_nodes_from(range(len(points)))
    for i in range(len(points) - 1):
        G.add_edge(i, i + 1)

    nx.draw_networkx_edges(
        G,
        points,
        ax=ax,
        width=1.0,
        edge_color=edge_color,
        alpha=alpha,
    )

    nx.draw_networkx_nodes(
        G,
        points,
        ax=ax,
        node_size=node_size,
        node_color=node_color,
        alpha=alpha,
    )


def draw_plan(
    ax,
    points_a,
    points_b,
    gamma,
    cmap="magma",
    **kwargs,
):
    """
    Draw the optimal transport plan.
    args:
        ax (matplotlib.axes.Axes): the axis to draw on.
        points_a (N, 2): the first set of points.
        points_b (M, 2): the second set of points.
        gamma (N, M): the optimal transport plan.
        cmap (str): the colormap to use.
    """
    assert points_a.shape[0] == gamma.shape[0], f"{points_a.shape}, {gamma.shape}"
    assert points_b.shape[0] == gamma.shape[1], f"{points_b.shape}, {gamma.shape}"
    G_gamma = _weighted_bipartite_graph(gamma)
    pos = np.concatenate([points_a, points_b], axis=0)

    _draw_weighted_graph(ax, G_gamma, pos, edge_cmap=cmap, edge_cbar_label="Weight", **kwargs)
    ax.set_aspect("equal")


def render_plans(
    points_a: list[np.ndarray],
    points_b: list[np.ndarray],
    coupling: list[np.ndarray],
    weights: list[np.ndarray],
    cmap="magma",
) -> list[np.ndarray]:
    """
    Render the optimal transport plan.
    args:
        points_a (N, 2): the first set of points.
        points_b (M, 2): the second set of points.
        coupling (N, M): the optimal transport plan.
        weights (N, M): the weights of the coupling.
        cmap (str): the colormap to use.
    """

    min_x = -0.1 + min([min(p[:, 0]) for p in points_a + points_b])
    max_x = 0.1 + max([max(p[:, 0]) for p in points_a + points_b])
    min_y = -0.1 + min([min(p[:, 1]) for p in points_a + points_b])
    max_y = 0.1 + max([max(p[:, 1]) for p in points_a + points_b])

    # pas = np.concatenate(points_a, axis=0)
    # pa_idx = np.unique(pas, axis=0, return_index=True)[1]
    # pa_idx = np.sort(pa_idx)
    # all_pas = pas[pa_idx]
    pbs = np.concatenate(points_b, axis=0)
    pb_idx = np.unique(pbs, axis=0, return_index=True)[1]
    pb_idx = np.sort(pb_idx)
    all_pbs = pbs[pb_idx]

    def draw_frame(pa, pb, c, w, t=None):
        fig, (ax, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        draw_plan(ax, pa, pb, c, cmap=cmap, node_size=10)
        # _draw_line_graph(ax, all_pas, alpha=0.2, node_color="blue", edge_color="blue", node_size=15)
        _draw_line_graph(ax, all_pbs, alpha=0.2, node_color="green", edge_color="green", node_size=15)
        ax.set_xlim(min_x, max_x)
        ax.set_ylim(min_y, max_y)
        ax.grid(True)
        ax.set_title("Coupling")

        ax2.imshow(w, cmap=cmap)
        ax2.set_title("Weights")
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=w.min(), vmax=w.max()))
        sm.set_array([])
        plt.colorbar(sm, ax=ax2, label="Weight")

        ax3.imshow(c, cmap=cmap)
        ax3.set_title("Coupling")
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize())
        sm.set_array([])
        plt.colorbar(sm, ax=ax3, label="Weight")

        if t is not None:
            fig.suptitle(f"time={t}")
        return fig2data(fig)

    return np.stack(
        [
            draw_frame(pa, pb, c, w, t)
            for pa, pb, c, w, t in tqdm(
                zip(points_a, points_b, coupling, weights, range(len(coupling))),
                desc="Rendering plan",
                total=len(coupling),
            )
        ]
    )
