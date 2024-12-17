import torch


def compute_global_nodes_transforms(local_nodes_transforms, nodes_indices, nodes_parents_list):
    """
    Follows the node hierarchy to compute the global node transforms in PyTorch.

    Parameters:
    local_nodes_transforms: torch.Tensor
        Local transformations of nodes, shape (num_nodes, 4, 4)
    nodes_indices: list of int
        Indices of nodes for which to compute the global transformations.
    nodes_parents_list: list of list of int
        Each sublist contains the indices of parent nodes for the corresponding node.
    
    Returns:
    global_nodes_transforms: torch.Tensor
        Global transformations of nodes, shape (len(nodes_indices), 4, 4)
    """
    global_nodes_transforms = []
    for node_index in nodes_indices:
        global_node_transform = local_nodes_transforms[node_index]
        # Iterate over all parents and multiply their matrices to the current node matrix
        for parent_index in nodes_parents_list[int(node_index)]:
            parent_matrix = local_nodes_transforms[parent_index]
            global_node_transform = torch.matmul(global_node_transform, parent_matrix)
        global_nodes_transforms.append(global_node_transform)
    
    # Stack the list of tensors into a single tensor
    global_nodes_transforms = torch.stack(global_nodes_transforms)
    
    return global_nodes_transforms


def compute_global_nodes_transforms_batched(local_nodes_transforms, nodes_indices, nodes_parents_list):
    global_nodes_transforms = []
    for _local_nodes_transforms, _nodes_indices, _nodes_parents_list in zip(local_nodes_transforms, nodes_indices, nodes_parents_list):
        global_nodes_transforms.append(compute_global_nodes_transforms(_local_nodes_transforms, _nodes_indices, _nodes_parents_list))
    return torch.stack(global_nodes_transforms)


def compute_skinning_matrices(global_joint_transforms, inverse_bind_matrices, joints, weights):
    """
    Computes skinning matrices for each vertex, with an optional batch dimension.
    
    Parameters:
    global_joint_transforms: torch.Tensor of shape (batch_size, num_joints, 4, 4) or (num_joints, 4, 4)
        Global joint transformations for each joint.
    inverse_bind_matrices: torch.Tensor of shape (batch_size, num_joints, 4, 4) or (num_joints, 4, 4)
        Inverse bind matrices for each joint.
    joints: torch.Tensor of shape (batch_size, num_vertices, 4) or (num_vertices, 4)
        Indices of joints affecting each vertex.
    weights: torch.Tensor of shape (batch_size, num_vertices, 4) or (num_vertices, 4)
        Weights of the influence of each joint on each vertex.

    Returns:
    skinning_matrices: torch.Tensor of shape (batch_size, num_vertices, 4, 4) or (num_vertices, 4, 4)
        The computed skinning matrices for each vertex.
    """
    # Detect if the input is batched
    is_batched = global_joint_transforms.dim() == 4
    
    # Ensure inputs are batched
    if not is_batched:
        global_joint_transforms = global_joint_transforms.unsqueeze(0)
        inverse_bind_matrices = inverse_bind_matrices.unsqueeze(0)
        joints = joints.unsqueeze(0)
        weights = weights.unsqueeze(0)
    
    # Compute Joint Matrices by batch matrix multiplication
    joint_matrices = torch.matmul(inverse_bind_matrices, global_joint_transforms)
    
    # Expand and prepare joints for batched indexing
    batch_size, num_vertices = joints.shape[:2]
    batch_indices = torch.arange(batch_size, device=joints.device)[:, None, None].expand(-1, num_vertices, 4)
    
    # Select and weight joint matrices
    selected_joint_matrices = joint_matrices[batch_indices, joints]
    weighted_joint_matrices = selected_joint_matrices * weights[..., None, None]
    skinning_matrices = weighted_joint_matrices.sum(dim=2)  # Sum over the joint dimension
    
    # Remove the batch dimension if it was added
    if not is_batched:
        skinning_matrices = skinning_matrices.squeeze(0)
    
    return skinning_matrices


def transform_vertices(vertices, skinning_matrices):
    """
    Transforms vertices using the given skinning matrices.
    
    Parameters:
    vertices: torch.Tensor of shape (..., num_vertices, 3)
    skinning_matrices: torch.Tensor of shape (..., num_vertices, 4, 4)
    
    Returns:
    transformed_vertices: torch.Tensor of shape (..., num_vertices, 3)
    """
    # Expand dimensions of vertices to (..., num_vertices, 1, 4) for matmul, by adding homogeneous coordinate
    ones = torch.ones(vertices.shape[:-1] + (1,), device=vertices.device, dtype=vertices.dtype)
    vertices_homogeneous = torch.cat([vertices, ones], dim=-1).unsqueeze(-2)
    
    # Perform batch matrix multiplication and then remove the added dimension
    # The result has shape (..., num_vertices, 1, 4), then we squeeze the second to last dim
    transformed_vertices_homogeneous = torch.matmul(vertices_homogeneous, skinning_matrices)
    
    # Slice to remove the homogeneous coordinate and get the final transformed vertices
    transformed_vertices = transformed_vertices_homogeneous[..., 0, :3]
    
    return transformed_vertices


def skin_mesh(vertices, vertex_joints, vertex_weights, joints, inverse_bind_matrices, local_nodes_transforms, nodes_parents_list):
    # Compute the global joint transforms
    # This involves traversing the scene graph and computing the global transform of each joint node
    # The global transform of a joint node is the product of the local transform of the joint node and the global transform of its parent joint node.
    # TODO: nicer handling of batched inputs
    if vertices.dim() == 3:
        global_joint_transforms = compute_global_nodes_transforms_batched(local_nodes_transforms, joints, nodes_parents_list)
    else:
        global_joint_transforms = compute_global_nodes_transforms(local_nodes_transforms, joints, nodes_parents_list)

    # Compute the skinning matrices
    skinning_matrices = compute_skinning_matrices(global_joint_transforms, inverse_bind_matrices, vertex_joints, vertex_weights)

    # Transform the vertices
    transformed_vertices = transform_vertices(vertices, skinning_matrices)
    
    aux = {
        "global_joint_transforms": global_joint_transforms,
        "skinning_matrices": skinning_matrices
    }

    return transformed_vertices, aux
