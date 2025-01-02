import numpy as np
import plotly.graph_objects as go
from ipywidgets import widgets, GridBox, Layout
from IPython.display import display

from .skin import transform_vertices
from .visuals_2d import plot_mesh_2d
from .visuals_mpl import plot_mesh_3d as plot_mesh_3d_mpl

import torch


def create_bone_mesh(global_joint_transforms):
    # create a trianlge representing the bone
    corners = 0.05
    bone_vertices_single = np.array(
        [[0, 0, 0], # bottom
         # four corners of the top
         [-corners, 0.5, -corners], # bottom left
         [-corners, 0.5, corners], # bottom right
         [corners, 0.5, corners], # top right
         [corners, 0.5, -corners]]) # top left
    # create indices for the bone
    # four triangles from the bottom to one of the top corners
    # two triangles for the top
    bone_indices_single = np.array([
        [0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 1], [1, 2, 3], [1, 3, 4]
        ])
    # repeat the bone for each joint
    n_bone_vertices = len(bone_vertices_single)
    bone_vertices = np.tile(bone_vertices_single, (len(global_joint_transforms), 1))
    # create indices for the bone
    # need to repeat the indices for each bone and increment the indices
    bone_indices = np.tile(bone_indices_single, (len(global_joint_transforms), 1))
    bone_indices += np.repeat(np.arange(0, len(global_joint_transforms) * n_bone_vertices, n_bone_vertices), len(bone_indices_single)).reshape(-1, 1)
    # each bone is a triangle, so we need to repeat global_joint_transforms
    global_joint_transforms = np.repeat(global_joint_transforms, n_bone_vertices, axis=0)
    # transform the bone to the correct position
    bone_vertices = transform_vertices(torch.Tensor(bone_vertices), torch.Tensor(global_joint_transforms)).numpy()
    return bone_vertices, bone_indices


# Changed opacity to 0.0 from 0.50
def plot_mesh_3d(vertices, indices, xlim=None, ylim=None, zlim=None, color='grey', facecolor=None, plot_vertices=True, fig=None, markersize=2, linewidth=1, opacity=0.40, name=None):
    # Ensure vertices is a numpy array for easier manipulation
    vertices = np.asarray(vertices)
    indices = np.asarray(indices)

    # Adjust vertices to swap Y and Z to make Y the up-axis
    vertices = vertices.copy()
    vertices[:, [1, 2]] = vertices[:, [2, 1]]
    
    # Extract vertices positions
    x, y, z = vertices[:, 0], vertices[:, 1], vertices[:, 2]
    
    # Extract the indices for the vertices of each face
    i, j, k = indices[:, 0], indices[:, 1], indices[:, 2]
    
    # Create the 3D mesh plot
    mesh = go.Mesh3d(x=x, y=y, z=z, i=i, j=j, k=k, color=color, opacity=opacity, flatshading=True, name=name)
    
    data = [mesh]
    
    # If plot_vertices is True, add the vertices as a scatter plot
    if plot_vertices:
        vertices_name = name + "_vertices" if name is not None else "vertices"
        vertices_plot = go.Scatter3d(x=x, y=y, z=z, mode='markers', marker=dict(size=markersize, color='red'), name=vertices_name)
        data.append(vertices_plot)
        
    
    # Create the figure
    if fig is None:
        fig = go.FigureWidget(data=data)
    else:
        for d in data:
            fig.add_trace(d)
    fig.update_layout(
                    showlegend=True,
                    scene=dict(
                        xaxis_title='X Axis',
                        yaxis_title='Z Axis',
                        zaxis_title='Y Axis (up)'),
                        margin=dict(r=20, l=10, b=10, t=10))
    fig.update_layout(width=800, height=500)  # Adjust these values as needed
    fig.update_scenes(aspectmode='data') # Ensures same scaling on all axes
    # Update layout to reverse z-axis
    fig.update_layout(scene=dict(yaxis=dict(autorange='reversed')))

    return fig


def update_mesh_3d(fig, data_indexes, new_data):
    for data_index, new_data_element in zip(data_indexes, new_data):
        data_element = fig.data[data_index]
        data_element.x = new_data_element[:, 0]
        data_element.y = new_data_element[:, 2]
        data_element.z = new_data_element[:, 1]
    fig.update_traces()
    return fig


def update_skinned_mesh_3d(fig, vertices, global_joint_transforms):
    bone_meshes = [create_bone_mesh(x[None])[0] for x in global_joint_transforms]
    # FIXME: Assumes that the fig.data elements are in the order: vertices (mesh), vertices (scatter), bone meshes
    # Updates them in reverse order for nicer visualization
    data_indexes = list(range(2, len(fig.data))) + [0, 1]
    _ = update_mesh_3d(fig, data_indexes, bone_meshes + [vertices, vertices])


def visualize_bones(global_joint_transforms, dims=[0, 1, 2], ax=None, in3d=False, plotly=True, name=None):
    bone_vertices, bone_indices = create_bone_mesh(global_joint_transforms)
    # plot the bone
    if in3d:
        if plotly:
            return plot_mesh_3d(bone_vertices, bone_indices, color='green', plot_vertices=False, opacity=1.0, fig=ax, name=name)
        else:
            return plot_mesh_3d_mpl(bone_vertices, bone_indices, color='green', plot_vertices=True, ax=ax)
    else:
        return plot_mesh_2d(bone_vertices[:, dims], bone_indices, color='green', facecolor='green', plot_vertices=True, ax=ax)


def visualize_bones_individualy(global_joint_transforms, dims=[0, 1, 2], ax=None, in3d=False, plotly=True, names=None):
    global_joint_transforms = np.asarray(global_joint_transforms)
    if names is None:
        names = [None] * len(global_joint_transforms)
    for global_joint_transform, name in zip(global_joint_transforms, names):
        ax = visualize_bones(global_joint_transform[None], dims=dims, ax=ax, in3d=in3d, plotly=plotly, name=name)
    return ax


def add_visibility_control_for_fig_data(fig):
    # Use checkboxes to toggle the visibility of each trace interactively
    checkboxes = [widgets.Checkbox(value=True, description=data.name) for data in fig.data]
    # ui = widgets.VBox(children=checkboxes)
    # Define the layout for the GridBox
    grid_layout = Layout(grid_template_columns='repeat(5, 150px)',  # Adjust the number of columns and width as needed
                        grid_gap='5px')  # Adjust the gap between checkboxes as needed

    # Use GridBox to organize checkboxes in a table-like layout
    ui = GridBox(children=checkboxes, layout=grid_layout)

    def update_traces(change):
        for i, checkbox in enumerate(checkboxes):
            fig.data[i].visible = checkbox.value

    # Attach callbacks to the checkboxes
    for checkbox in checkboxes:
        checkbox.observe(update_traces, names='value')

    # Display UI and figure
    display(ui)
    display(fig)


def plot_skinned_mesh_3d(vertices, faces, global_joint_transforms, bone_names=None, visibility_control=False):
    fig = plot_mesh_3d(vertices, faces, plot_vertices=False, name="mesh")
    # fig = visualize_bones_individualy(global_joint_transforms, dims=[2, 1], ax=fig, in3d=True, names=bone_names)
    if visibility_control:
        add_visibility_control_for_fig_data(fig)
    return fig
