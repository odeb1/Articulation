from pygltflib import GLTF2
import numpy as np
import dos.skingltf.gltf
import dos.skingltf.skin
import dos.skingltf.visuals
from dataclasses import dataclass
import torch
from einops import repeat


@dataclass
class Skin(object):
    # Data class attribute to hold the file path of the glTF file.
    gltf_file_path: str

    def __post_init__(self):
        # This method is automatically called after the data class has been initialized.
        # Load the glTF file specified by the path provided.
        self.gltf = GLTF2().load(self.gltf_file_path)
        
        # Read the skinned mesh data from the glTF object.
        mesh = dos.skingltf.gltf.read_skinned_mesh(self.gltf)
        
        # Extract mesh attributes like vertices, faces, joints, etc., from the skinned mesh data.
        self.vertices = mesh["vertices"]
        self.faces = mesh["faces"]
        self.joints = mesh["joints"]
        self.vertex_joints = mesh["vertex_joints"]
        self.vertex_weights = mesh["vertex_weights"]
        self.inverse_bind_matrices = mesh["inverse_bind_matrices"]
        self.nodes_parents_list = mesh["nodes_parents_list"]
        self.local_nodes_transforms = mesh["local_nodes_transforms"]
        self.node_names = mesh["node_names"]

        # Print out a message indicating that the skinned mesh has been successfully loaded.
        print(f"Loaded skinned mesh with {len(self.local_nodes_transforms)} nodes.")

    def skin_mesh(self, vertices, local_nodes_transforms):
        # This function skins the mesh using the input vertices and local node transforms.
        
        # check vertices are the same, compare values (close enough)
        # TODO: load mesh from the same file
        # import lovely_tensors as lt; import ipdb; lt.monkey_patch(); ipdb.set_trace()  # fmt: skip

        # Get the device type of the input vertices (e.g., CPU or GPU).
        device = vertices.device
        # FIXME: assert
        # assert torch.allclose(vertices, self.vertices.to(device)[None])
        
        # Repeating the vertex joints, weights, joints, and other data across batch dimension ("b").
        # This is done to match the batch size of the vertices.
        b = vertices.shape[0]

        # Call the function to skin the mesh using the attributes of the class.
        # The repeat operation makes sure each batch gets the same joint and weight data.
        return dos.skingltf.skin.skin_mesh(
            vertices,
            repeat(self.vertex_joints, "... -> b ...", b=b).to(device),
            repeat(self.vertex_weights, "... -> b ...", b=b).to(device),
            repeat(self.joints, "... -> b ...", b=b).to(device),
            repeat(self.inverse_bind_matrices, "... -> b ...", b=b).to(device),
            local_nodes_transforms.to(device),
            [self.nodes_parents_list] * b,  # Repeat the nodes parents list across the batch dimension.
        )

    def find_root_bone_index(self, nodes_parents_list):
        # Iterate through the dictionary and check for empty lists
        root_bone_index = [key for key, value in nodes_parents_list.items() if not value]
        return root_bone_index

    def skin_mesh_with_rotations(self, vertices, rotations, root_bone_index=43):
        """
        Skin the mesh with a set of input rotations for each bone.
        rotations: (B, num_bones, 4)
        """
        # Determine batch size.
        b = vertices.shape[0]

        # Obtain the initial local node transformations and move them to the same device as vertices.
        local_nodes_transforms = self.local_nodes_transforms.to(vertices.device)

        # Repeat local node transforms across batch dimension to match input vertices.
        local_nodes_transforms = repeat(local_nodes_transforms, "... -> b ...", b=b)

        # Convert quaternion rotations to rotation matrices.
        rotation_matrices = dos.skingltf.geometry.quaternion_to_matrix(rotations).to(vertices.device)

        # Rotate the first three columns (the sub-matrix representing rotation components) of each transform matrix.
        rotated_submatrices = torch.matmul(rotation_matrices, local_nodes_transforms[:, :, :3, :3])

        # Handle root bone separately
        for bone_idx in range(rotations.shape[1]):
            if bone_idx == root_bone_index:
                # Separate handling for root bone rotation
                root_rotation_matrix = rotation_matrices[:, bone_idx]
                # For the root bone, reduce the impact of the rotation for stability
                # root_rotation_matrix = 0.80 * root_rotation_matrix
                root_rotation_matrix = root_rotation_matrix
                rotated_submatrices[:, bone_idx] = torch.matmul(root_rotation_matrix, local_nodes_transforms[:, bone_idx, :3, :3])
            else:
                # General handling for other bones
                rotated_submatrices[:, bone_idx] = torch.matmul(rotation_matrices[:, bone_idx], local_nodes_transforms[:, bone_idx, :3, :3])

        # Construct a new tensor combining the rotated sub-matrix and unmodified translation parts.
        local_nodes_transforms = torch.cat([
            torch.cat([rotated_submatrices, local_nodes_transforms[:, :, :3, 3:]], dim=3),  # Concatenate rotation and translation
            local_nodes_transforms[:, :, 3:, :]  # Include the homogeneous coordinates
        ], dim=2)

        # # import ipdb; ipdb.set_trace()
        # # Skin the mesh
        # transformed_vertices, aux = self.skin_mesh(vertices, local_nodes_transforms)
        # global_joint_transforms = aux["global_joint_transforms"]     
        # # Error:: RuntimeError: The size of tensor a (5) must match the size of tensor b (39) at non-singleton dimension 1   
        # fig = self.plot_skinned_mesh_3d(transformed_vertices.cpu().detach(), global_joint_transforms.cpu().detach())
        # fig.write_image("bones_img.png")
        
        # Skin the mesh using the modified transformations.
        return self.skin_mesh(vertices, local_nodes_transforms)

    def plot_skinned_mesh_3d(self, vertices, global_joint_transforms):
        # Plot the skinned mesh using a 3D visualization.
        # This function uses an external library (dos.skingltf.visuals) to render the skinned mesh.
        
        # vertices shape is (1, 7483, 3)
        # self.faces shape is torch.Size([14216, 3])
        # global_joint_transforms shape is (1, 45, 4, 4)
        
        
        return dos.skingltf.visuals.plot_skinned_mesh_3d(
            vertices,  # The transformed vertices to be visualized.
            self.faces,  # Mesh faces that define how vertices are connected.
            global_joint_transforms,  # Transformations to be applied to each joint.
        )
    
    
    
    
    