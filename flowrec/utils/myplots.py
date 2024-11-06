'''Some useful plots'''

import numpy as np
import matplotlib.colors as mcolors

from functools import partial
from typing import Optional

# user-defined color maps
discrete_dict = {
    'defne': ["#03BDAB", "#FEAC16", "#5D00E6","#F2BCF3","#AFEEEE"],
    'overleaf-earth': ['#1e446b','#26aa08','#bf9000','#ffc000'],
    'dunlop-etal-2024': ['#FAF3DD', '#B9D5BA', '#467F79', '#2E4F4A', '#F8DDDA', '#E1D7D0'],
    'cervia-etal-2024': ['#f7a24f', '#c6133b', '#90162d', '#93a5cb'],
    'wang-etal-2024': ['#7e4909', '#e5cc8f', '#0e8585', '#cce5e5', '#830783', '#e5cce5'],
    'bubblegum': [[87./256,212./256,254./256], [255./256,222./256,117./256], [225./256,178./256,239./256], [196./256,224./256,178./256], [255./256,188./256,193./256]],
    'trafficlight': ['#f18c25', '#d8614f', '#716ea9', '#6a9ace', '#1eb03d'],
    'trafficlight-pale': ['#faeca8', '#fdd5c0', '#cac0e1', '#abdaec', '#97d1a0'],
}
continuous_dict = {
    'defne': ["#03BDAB", "#FEAC16", "#5D00E6"],
    'overleaf-earth': ['#1e446b','#26aa08','#bf9000','#ffc000'],
    'cervia-etal-2024': ['#f7a24f', '#c6133b', '#90162d', '#93a5cb'],
    'pastel-blue': ['#bee8e8', '#4c9be6', '#4d4d9f'],
    'pastel-red': ['#f5dfd8', '#edb8b0', '#e69191', '#c25759'],
    'bubblegum': [[87./256,212./256,254./256], [255./256,222./256,117./256]],
}


def create_discrete_colormap(colors:list, name:str = 'custom_colormap'):
    """Create a discrete colormap from given color hex codes.
    
    Args:
        colors (list): List of color hex codes.
        name (str, optional): Name of the colormap. Defaults to 'custom_colormap'.
    
    Returns:
        matplotlib.colors.ListedColormap: The discrete colormap object.
    """
    cmap = mcolors.ListedColormap(colors, name=name)
    return cmap

def create_continuous_colormap(colors:list, name:str = 'custom_colormap', N:int = 256):
    """Create a continuous colormap from given color hex codes.
    
    Args:
        colors (list): List of color hex codes.
        name (str, optional): Name of the colormap. Defaults to 'custom_colormap'.
        N (int, optional): Number of color levels. Defaults to 256.
    
    Returns:
        matplotlib.colors.ListedColormap: The continuous colormap object.
    """
    ncolors = len(colors)
    if ncolors < 2:
        raise ValueError("Please provide at least two colors.")

    color_array = np.zeros((N, 4))
    for i in range(N):
        idx1 = int(i * (ncolors - 1) / N)
        idx2 = min(idx1 + 1, ncolors - 1)
        t = i * (ncolors - 1) / N - idx1
        color_array[i] = tuple((1 - t) * c1 + t * c2 for c1, c2 in zip(mcolors.to_rgba(colors[idx1]), mcolors.to_rgba(colors[idx2])))
    cmap = mcolors.ListedColormap(color_array, name=name)
    return cmap

def create_custom_colormap(map_name:str = 'defne',type:str = 'discrete', colors:Optional[list] = None, N:int = 256):
    """Create a custom colormap.

    This function creates either a discrete or continuous colormap based on the given parameters.

    Args:
        map_name (str, optional): Name of the custom colormap. Defaults to 'defne'.
        cmap_type (str, optional): Type of the colormap ('discrete' or 'continuous'). Defaults to 'discrete'.
        colors (list, optional): List of color hex codes. If None, uses predefined colormap based on map_name. Defaults to None.
        N (int, optional): Number of color levels for continuous colormap. Defaults to 256.

    Returns:
        matplotlib.colors.ListedColormap: The custom colormap object.
    """
    colors_dict = {'discrete': discrete_dict, 'continuous': continuous_dict}
    function_dict = {'discrete': create_discrete_colormap, 'continuous': partial(create_continuous_colormap, N=N)}
    if colors:
        assert isinstance(colors, list)
        colors_hex = colors
    else:
        try:
            colors_hex = colors_dict[type][map_name]
        except KeyError:
            print(f'map {map_name} does not exist.')
            raise NotImplementedError
    cmap_fn = function_dict[type]
    cmap = cmap_fn(colors_hex, map_name)
    return cmap


truegrey = '#808080'