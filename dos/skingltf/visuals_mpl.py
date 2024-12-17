import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection



def plot_mesh_3d(vertices, indices, xlim=None, ylim=None, zlim=None, color='red', facecolor='cyan', plot_vertices=True, ax=None, markersize=4, linewidth=1):
    # Ensure vertices is a numpy array for easier manipulation
    vertices = np.asarray(vertices)

    ax_created = False
    if ax is None:
        ax_created = True
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    
    # Create a Poly3DCollection object from the faces
    poly3d = [vertices[triangle] for triangle in indices]
    ax.add_collection3d(Poly3DCollection(poly3d, facecolors=facecolor, linewidths=linewidth, edgecolors=color, alpha=.2))

    # if plot_vertices:
    #     # Plot the vertices as scatter points
    #     ax.scatter(vertices[:,0], vertices[:,1], vertices[:,2], color=color, s=markersize)
    
    # # Set the limits
    # if xlim is None:
    #     xlim = (vertices[:,0].min(), vertices[:,0].max())
    # if ylim is None:
    #     ylim = (vertices[:,1].min(), vertices[:,1].max())
    # if zlim is None:
    #     zlim = (vertices[:,2].min(), vertices[:,2].max())
        
    lim = np.array([vertices.min(), vertices.max()])

    # ax.set_xlim(lim)
    # ax.set_ylim(lim)
    # ax.set_zlim(lim)
    ax.set_xlabel('X axis label')
    ax.set_ylabel('Y axis label')
    ax.set_zlabel('Z axis label')

    # Auto scale to the mesh size
    scale = vertices.flatten()
    ax.auto_scale_xyz(scale, scale, scale)

    return ax
