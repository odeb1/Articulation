from pygltflib import GLTF2
import numpy as np
import skingltf.gltf
import skingltf.skin
import skingltf.visuals
from dataclasses import dataclass
import torch
from einops import repeat

@dataclass
class Skin(object):
    gltf_file_path: str

    def __post_init__(self):
        # Load the glTF file
        gltf = GLTF2().load(self.gltf_file_path)
        mesh = skingltf.gltf.read_skinned_mesh(gltf)
        # Extract mesh data
        self.vertices = mesh["vertices"]
        self.faces = mesh["faces"]
        self.joints = mesh["joints"]
        self.vertex_joints = mesh["vertex_joints"]
        self.vertex_weights = mesh["vertex_weights"]
        self.inverse_bind_matrices = mesh["inverse_bind_matrices"]
        self.nodes_parents_list = mesh["nodes_parents_list"]
        self.local_nodes_transforms = mesh["local_nodes_transforms"]
        self.node_names = mesh["node_names"]

        print(f"Loaded skinned mesh with {len(self.local_nodes_transforms)} nodes.")
        

    def skin_mesh(self, vertices, local_nodes_transforms):
        # check vertices are the same, compare values (close enough)
        # TODO: load mesh from the same file
        # import lovely_tensors as lt; import ipdb; lt.monkey_patch(); ipdb.set_trace()  # fmt: skip
        device = vertices.device
        # FIXME: assert
        # assert torch.allclose(vertices, self.vertices.to(device)[None])
        b = vertices.shape[0]
        return skingltf.skin.skin_mesh(
            vertices,
            repeat(self.vertex_joints, "... -> b ...", b=b).to(device),
            repeat(self.vertex_weights, "... -> b ...", b=b).to(device),
            repeat(self.joints, "... -> b ...", b=b).to(device),
            repeat(self.inverse_bind_matrices, "... -> b ...", b=b).to(device),
            local_nodes_transforms.to(device),
            [self.nodes_parents_list] * b,
        )


    def skin_mesh_with_rotations(self, vertices, rotations):
        """
        rotations: (B, num_bones, 4) TODO: use quaternions
        """
        # adjust the initial local nodes transforms
        b = vertices.shape[0]
        local_nodes_transforms = self.local_nodes_transforms.to(vertices.device)
        local_nodes_transforms = repeat(local_nodes_transforms, "... -> b ...", b=b)
        rotation_matrices = skingltf.geometry.quaternion_to_matrix(rotations).to(vertices.device)
        # Perform matrix multiplication without in-place operations
        rotated_submatrices = torch.matmul(rotation_matrices, local_nodes_transforms[:, :, :3, :3])
        # Construct a new tensor that combines the results of the rotation with the unchanged parts of local_nodes_transforms
        local_nodes_transforms = torch.cat([
            torch.cat([rotated_submatrices, local_nodes_transforms[:, :, :3, 3:]], dim=3),
            local_nodes_transforms[:, :, 3:, :]
        ], dim=2)
        return self.skin_mesh(vertices, local_nodes_transforms)


    def plot_skinned_mesh_3d(self, vertices, global_joint_transforms):
        return skingltf.visuals.plot_skinned_mesh_3d(
            vertices,
            self.faces,
            global_joint_transforms,
        )
