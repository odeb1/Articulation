"""
Follows Pytorch3D conventions

.. code-block:: python

    M = [
            [Rxx, Ryx, Rzx, 0],
            [Rxy, Ryy, Rzy, 0],
            [Rxz, Ryz, Rzz, 0],
            [Tx,  Ty,  Tz,  1],
        ]

To apply the transformation to points, which are row vectors, the latter are
converted to homogeneous (4D) coordinates and right-multiplied by the M matrix:

.. code-block:: python

    points = [[0, 1, 2]]  # (1 x 3) xyz coordinates of a point
    [transformed_points, 1] ∝ [points, 1] @ M

See https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/transforms/transform3d.py
"""
import pygltflib
import numpy as np
from .geometry import quaternion_to_matrix_numpy


# https://github.com/KhronosGroup/glTF-Tutorials/issues/21#issuecomment-704437553
DATA_URI_HEADER_OCTET_STREAM = "data:application/octet-stream;base64,"
DATA_URI_HEADER_GLTF_BUFFER = "data:application/gltf-buffer;base64,"

# https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#accessor-data-types
NUMBER_OF_COMPONENETS = {
    "SCALAR": 1,
    "VEC2": 2,
    "VEC3": 3,
    "VEC4": 4,
    "MAT2": 4,
    "MAT3": 9,
    "MAT4": 16,
}
NUMBER_OF_BYTES = {
    5120: 1,
    5121: 1,
    5122: 2,
    5123: 2,
    5125: 4,
    5126: 4,
}
COMPONENT_TYPE = {
    5120: "BYTE",
    5121: "UNSIGNED_BYTE",
    5122: "SHORT",
    5123: "UNSIGNED_SHORT",
    5125: "UNSIGNED_INT",
    5126: "FLOAT",
}
COMPONENT_TYPE_TO_NUMPY = {
    "BYTE": np.int8,
    "UNSIGNED_BYTE": np.uint8,
    "SHORT": np.int16,
    "UNSIGNED_SHORT": np.uint16,
    "UNSIGNED_INT": np.uint32,
    "FLOAT": np.float32,
}

# https://github.com/KhronosGroup/glTF-Tutorials/issues/21#issuecomment-704437553
DATA_URI_HEADER_OCTET_STREAM = "data:application/octet-stream;base64,"
DATA_URI_HEADER_GLTF_BUFFER = "data:application/gltf-buffer;base64,"

# https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#accessor-data-types
NUMBER_OF_COMPONENETS = {
    "SCALAR": 1,
    "VEC2": 2,
    "VEC3": 3,
    "VEC4": 4,
    "MAT2": 4,
    "MAT3": 9,
    "MAT4": 16,
}
NUMBER_OF_BYTES = {
    5120: 1,
    5121: 1,
    5122: 2,
    5123: 2,
    5125: 4,
    5126: 4,
}
COMPONENT_TYPE = {
    5120: "BYTE",
    5121: "UNSIGNED_BYTE",
    5122: "SHORT",
    5123: "UNSIGNED_SHORT",
    5125: "UNSIGNED_INT",
    5126: "FLOAT",
}
COMPONENT_TYPE_TO_NUMPY = {
    "BYTE": np.int8,
    "UNSIGNED_BYTE": np.uint8,
    "SHORT": np.int16,
    "UNSIGNED_SHORT": np.uint16,
    "UNSIGNED_INT": np.uint32,
    "FLOAT": np.float32,
}



def read_data(gltf, accessor_id):
    """
    
    For more info, see https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#accessor-data-types
    """
    # Get accesor
    accessor = gltf.accessors[accessor_id]

    # Read joint indices data from the buffer view
    buffer_view = gltf.bufferViews[accessor.bufferView]
    buffer = gltf.buffers[buffer_view.buffer]

    # monkey patch header constant needed for gltf.get_data_from_buffer_uri
    if DATA_URI_HEADER_OCTET_STREAM in buffer.uri:
        pygltflib.DATA_URI_HEADER = DATA_URI_HEADER_OCTET_STREAM
    elif DATA_URI_HEADER_GLTF_BUFFER in buffer.uri:
        pygltflib.DATA_URI_HEADER = DATA_URI_HEADER_GLTF_BUFFER
    elif "9abfce885a834399b2c3ccaed51cd474.bin" in buffer.uri:
        pygltflib.DATA_URI_HEADER = "9abfce885a834399b2c3ccaed51cd474.bin"
    elif "9abfce885a834399b2c3ccaed51cd471-blender-quads.bin" in buffer.uri:
        pygltflib.DATA_URI_HEADER = "9abfce885a834399b2c3ccaed51cd471-blender-quads.bin"
    elif "SheepFemaleRD.bin" in buffer.uri:
        pygltflib.DATA_URI_HEADER = "SheepFemaleRD.bin"
    elif "HorseFemaleRD.bin" in buffer.uri:
        pygltflib.DATA_URI_HEADER = "HorseFemaleRD.bin"
    elif "bonnie.bin" in buffer.uri:
        pygltflib.DATA_URI_HEADER = "bonnie.bin"
    elif "CowMaleRD" in buffer.uri:
        pygltflib.DATA_URI_HEADER = "CowMaleRD"
    elif "giraffe" in buffer.uri:
        pygltflib.DATA_URI_HEADER = "giraffe"
    elif "tiger" in buffer.uri:
        pygltflib.DATA_URI_HEADER = "tiger"
    elif "simpson" in buffer.uri:
        pygltflib.DATA_URI_HEADER = "simpson"
    elif "cat_bengal" in buffer.uri:
        pygltflib.DATA_URI_HEADER = "cat_bengal"
    elif "cat" in buffer.uri:
        pygltflib.DATA_URI_HEADER = "cat"
    elif "elephant" in buffer.uri:
        pygltflib.DATA_URI_HEADER = "elephant"
    elif "eagle" in buffer.uri:
        pygltflib.DATA_URI_HEADER = "eagle"
    elif "seagull" in buffer.uri:
        pygltflib.DATA_URI_HEADER = "seagull"
    elif "humming_bird" in buffer.uri:
        pygltflib.DATA_URI_HEADER = "humming_bird"
    elif "fish" in buffer.uri:
        pygltflib.DATA_URI_HEADER = "fish"
    elif "frog" in buffer.uri:
        pygltflib.DATA_URI_HEADER = "frog"
    elif "dog" in buffer.uri:
        pygltflib.DATA_URI_HEADER = "dog"
    elif "lion" in buffer.uri:
        pygltflib.DATA_URI_HEADER = "lion"
    elif "polar_bear" in buffer.uri:
        pygltflib.DATA_URI_HEADER = "polar_bear"
    elif "cartoon_monkey" in buffer.uri:
        pygltflib.DATA_URI_HEADER = "cartoon_monkey"
    elif "butterfly" in buffer.uri:
        pygltflib.DATA_URI_HEADER = "butterfly"
    elif "rabbit" in buffer.uri:
        pygltflib.DATA_URI_HEADER = "rabbit"
    elif "alpaca" in buffer.uri:
        pygltflib.DATA_URI_HEADER = "alpaca"
    elif "penguin" in buffer.uri:
        pygltflib.DATA_URI_HEADER = "penguin"
    elif "yellow_bird" in buffer.uri:
        pygltflib.DATA_URI_HEADER = "yellow_bird"
    elif "bird" in buffer.uri:
        pygltflib.DATA_URI_HEADER = "bird"
    else:
        print(f"Debug: Unknown data URI header found in buffer.uri: {buffer.uri}")
        raise ValueError("Unknown data uri header")
    
    # read data as bytes
    data = gltf.get_data_from_buffer_uri(buffer.uri)
    
    # decode data to numpy array
    data_offset = buffer_view.byteOffset + accessor.byteOffset
    element_size = NUMBER_OF_COMPONENETS[accessor.type] * NUMBER_OF_BYTES[accessor.componentType]
    if buffer_view.byteStride is None:
        step = element_size
    else:
        step = buffer_view.byteStride
    array = []
    for i in range(accessor.count):
        start = data_offset + i * step
        end = start + element_size
        array += [np.frombuffer(data[start:end], dtype=COMPONENT_TYPE_TO_NUMPY[COMPONENT_TYPE[accessor.componentType]])]
    array = np.stack(array)

    return array


def get_node_transform(node):
    """
    Convention followed by glTF (also Pytorch3D and OpenGL?):

        M = [
            [Rxx, Ryx, Rzx, 0],
            [Rxy, Ryy, Rzy, 0],
            [Rxz, Ryz, Rzz, 0],
            [Tx,  Ty,  Tz,  1],
        ]
    """
    if node.matrix is not None:
        return np.array(node.matrix).reshape(4, 4)
    else:
        # compute the transform from translation, rotation and scale
        transform = np.eye(4)
        # first scale, then rotate, then translate
        if node.scale is not None:
            scale = np.array(node.scale)
            transform = transform @ np.diag(np.concatenate([scale, np.ones(1, dtype=scale.dtype)]))
        if node.rotation is not None:
            # rotation is a quaternion
            rotation_matrix = quaternion_to_matrix_numpy(node.rotation)
            transform = transform @ rotation_matrix
        if node.translation is not None:
            translation_matrix = np.eye(4)
            # column-major order
            translation_matrix[3, :3] = node.translation
            transform = transform @ translation_matrix
        return transform


def get_local_nodes_transforms(nodes):
    transforms = np.zeros((len(nodes), 4, 4), dtype=np.float32)
    for node_index, node in enumerate(nodes):
        transforms[node_index] = get_node_transform(node)
    return transforms


def get_nodes_parents(nodes):
    """
    Traverse the scene graph and list all the parents of each node.

    nodes: list of gltf nodes
    
    returns: dictionary where keys are nodes and values are their parents (indices)
    """

    # Create a dictionary where keys are nodes and values are their parents
    nodes_parents = {}
    
    # Recursively traverse the scene graph and fill the nodes_parents dictionary
    def traverse_scene_graph(node_index, parent):
        # If the node was already visited and has a parent, return
        if node_index in nodes_parents and nodes_parents[node_index] is not None:
            return
        nodes_parents[node_index] = parent
        for child in nodes[node_index].children:
            traverse_scene_graph(child, node_index)

    # Start the traversal from the root node (we don't know the root node, so we do it for all nodes that were not visited yet)
    for node_index in range(len(nodes)):
        traverse_scene_graph(node_index, None)

    # Now for each node create a list of all its parents
    nodes_parents_list = {}
    for node_index, parent in nodes_parents.items():
        nodes_parents_list[node_index] = []
        while parent is not None:
            nodes_parents_list[node_index].append(parent)
            parent = nodes_parents[parent]

    return nodes_parents_list


def read_skinned_mesh(gltf, pytorch=True):
    """
    Skin defines skeleton using joints and inverse bind matrices.
    Joints are indices to the nodes in the scene.
    Inverse bind matrices are indices to the accessor and they define the inverse of the global joint transform in its initial position.

    The vertices have to be transformed with the current global transform of the joint node.

    IMPORTANT: Vertex skinning in other contexts often involves a matrix that is called “Bind Shape Matrix”. This matrix is supposed to transform the geometry of the skinned mesh into the coordinate space of the joints. In glTF, this matrix is omitted, and it is assumed that this transform is either premultiplied with the mesh data, or postmultiplied to the inverse bind matrices. Source: https://github.khronos.org/glTF-Tutorials/gltfTutorial/gltfTutorial_020_Skins.html#vertex-skinning-implementation

    TODO: transform the inital mesh based on its the parent nodes or ensure that the transform is zero
    """
        
    assert len(gltf.meshes) == 1, "Only one mesh is supported."
    assert len(gltf.skins) == 1, "Only one skin is supported."

    # TODO: make this automatic and support multiple meshes and skins? Horse_stallion_IP does not need it
    mesh_index = 0
    skin_index = 0 
    mesh = gltf.meshes[mesh_index]
    skin = gltf.skins[skin_index]

    # TODO: potenitally support multiple primitives. When this is needed? Horse_stallion_IP does not need it
    assert len(mesh.primitives) == 1, "Only one primitive is supported."
    primitive = mesh.primitives[0]

    # Read triangle indices
    assert primitive.mode == 4, "Only triangles are supported."
    indices = read_data(gltf, primitive.indices)
    indices = indices.reshape(-1, 3)

    # Read attributes
    attributes = primitive.attributes

    # assert that is has only one attribute matching JOINTS_* and WEIGHTS_*
    assert len([k for k in dir(attributes) if "JOINTS_" in k]) == 1, "Only one attribute matching JOINTS_* is supported."
    assert len([k for k in dir(attributes) if "WEIGHTS_" in k]) == 1, "Only one attribute matching WEIGHTS_* is supported."

    # Read vertices, joint indices and weights
    vertices = read_data(gltf, attributes.POSITION)
    vertex_joints = read_data(gltf, attributes.JOINTS_0)
    vertex_weights = read_data(gltf, attributes.WEIGHTS_0)

    # Read inverse bind matrices
    inverse_bind_matrices = read_data(gltf, skin.inverseBindMatrices)
    # Note: matrices are column-major (traslation is the last row)
    # e.g.: 
    #     2.0,    0.0,    0.0,    0.0,
    #     0.0,    0.866,  0.5,    0.0,
    #     0.0,   -0.25,   0.433,  0.0,
    #    10.0,   20.0,   30.0,    1.0
    inverse_bind_matrices = inverse_bind_matrices.reshape(-1, 4, 4)

    # Traverse the scene graph and get the parents of each node
    # glTF indicates only childern, so we need to traverse the scene graph from the root node
    nodes_parents_list = get_nodes_parents(gltf.nodes)
    
    # Compute the local joint transforms
    local_nodes_transforms = get_local_nodes_transforms(gltf.nodes)

    # If pytorch is True, convert to torch tensors
    joints = skin.joints
    if pytorch:
        import torch
        vertices = torch.tensor(vertices, dtype=torch.float32)
        indices = torch.tensor(indices.astype(np.int64), dtype=torch.int64)
        joints = torch.tensor(joints, dtype=torch.int64)
        # vertex_joints = torch.tensor(vertex_joints, dtype=torch.int64)
        vertex_joints = torch.tensor(vertex_joints.astype(np.int64), dtype=torch.int64)

        
        vertex_weights = torch.tensor(vertex_weights, dtype=torch.float32)
        inverse_bind_matrices = torch.tensor(inverse_bind_matrices, dtype=torch.float32)
        local_nodes_transforms = torch.tensor(local_nodes_transforms, dtype=torch.float32)

    return {
        "vertices": vertices,
        "faces": indices,
        "joints": joints,
        "vertex_joints": vertex_joints,
        "vertex_weights": vertex_weights,
        "inverse_bind_matrices": inverse_bind_matrices,
        "nodes_parents_list": nodes_parents_list,
        "local_nodes_transforms": local_nodes_transforms,
        "node_names": [node.name for node in gltf.nodes]
    }
        