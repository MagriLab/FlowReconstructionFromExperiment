'''Collection of useful functions'''

from .myplots import create_custom_colormap, truegrey
my_discrete_cmap = create_custom_colormap(map_name='overleaf-earth',type='discrete')
my_continuous_cmap = create_custom_colormap(map_name='overleaf-earth',type='continuous')