import matplotlib.pyplot as plt
from matplotlib.patches import Polygon


def plot_mesh_2d(vertices, indices, xlim=None, ylim=None, color='red', facecolor='none', plot_vertices=True, ax=None, markersize=4, linewidth=1):
    vertices = vertices[:, :2]
    # Plot the mesh (in 2D)

    ax_created = False
    if ax is None:
        ax_created = True
        fig, ax = plt.subplots()
    # Plot each face as a Polygon
    for face in indices:
        polygon = Polygon(vertices[face], closed=True, edgecolor=color, facecolor=facecolor, linewidth=linewidth)
        ax.add_patch(polygon)
    if plot_vertices:
        # Plot the vertices
        ax.plot(vertices[:,0], vertices[:,1], 'o', color=color, markersize=markersize)  

    if ax_created:
        if xlim is None:
                xlim = (vertices[:,0].min() - 0.1, vertices[:,0].max() + 0.1)
        if ylim is None:
                ylim = (vertices[:,1].min() - 0.1, vertices[:,1].max() + 0.1)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect('equal')

    return ax
