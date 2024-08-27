from typing import List
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from .rgb_ind_convertor import floorplan_fuse_map  # Make sure the import path is correct

LABEL_MAP = {
    'background': 0,
    'closet': 1,
    'bathroom': 2,
    'living room': 3,
    'kitchen': 4,
    'dining room': 5,
    'bedroom': 6,
    'hall': 7,
    'balcony': 8,
    'not used': 9,
    'door': 10,
    'window': 11,
    'wall': 12
}

def export_legend(
    legend: matplotlib.legend.Legend,
    filename: str = "legend.png",
    expand: List[int] = [-5, -5, 5, 5],
):
    fig = legend.figure
    fig.canvas.draw()
    bbox = legend.get_window_extent()
    bbox = bbox.from_extents(*(bbox.extents + np.array(expand)))
    bbox = bbox.transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(filename, dpi="figure", bbox_inches=bbox)

def norm255to1(x: List[int]) -> List[float]:
    return [p / 255 for p in x]

def handle(m: str, c: List[float]):
    return plt.plot([], [], marker=m, color=c, ls="none")[0]

def main():
    colors = [
        "background",
        "closet",
        "bathroom",
        "living room\nkitchen\ndining room",
        "bedroom",
        "hall",
        "balcony",
        "not used",
        "not used",
        "door",
        "window",
        "wall",
    ]
    
    colors2 = [norm255to1(rgb) for rgb in list(floorplan_fuse_map.values())]

    # Manually setting distinct colors for door and window
    door_color = [1.0, 0.0, 0.0]  # Example: Red
    window_color = [0.0, 1.0, 0.0]  # Example: Green
    colors2[LABEL_MAP['door']] = door_color
    colors2[LABEL_MAP['window']] = window_color

    handles = [handle("s", colors2[i]) for i in range(len(colors))]
    labels = colors
    legend = plt.legend(handles, labels, loc=3, framealpha=1, frameon=True)
    export_legend(legend)

if __name__ == "__main__":
    main()