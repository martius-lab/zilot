import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap

colors = [
    (0.368, 0.507, 0.71),
    (0.881, 0.611, 0.142),
    (0.56, 0.692, 0.195),
    (0.923, 0.386, 0.209),
    (0.528, 0.471, 0.701),
    (0.772, 0.432, 0.102),
    (0.364, 0.619, 0.782),
    (0.572, 0.586, 0.0),
]


OUR_BLUE = colors[0]
OUR_ORANGE = colors[1]
OUR_GREEN = colors[2]
OUR_RED = colors[3]
OUR_PURPLE = colors[4]
OUR_BROWN = colors[5]
OUR_LIGHTBLUE = colors[6]
OUR_DARKGREEN = colors[7]

TRAJ_COLOR = OUR_BLUE
GOAL_COLOR = OUR_ORANGE
COST_COLOR = OUR_GREEN
COUPLING_COLOR = OUR_RED
BASELINE_COLOR = OUR_LIGHTBLUE

TRAJ_CMAP = LinearSegmentedColormap.from_list("custom_white_to_color", ["white", OUR_BLUE])
GOAL_CMAP = LinearSegmentedColormap.from_list("custom_white_to_color", ["white", OUR_ORANGE])
COST_CMAP = cm.get_cmap("summer")
COUPLING_CMAP = LinearSegmentedColormap.from_list("custom_white_to_color", ["white", OUR_RED])


def darker(color, factor=0.87):
    r, g, b = color
    return (min(1, r * factor), min(1, g * factor), min(1, b * factor))
