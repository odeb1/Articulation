import functools
import math
import random
from dataclasses import dataclass

#import kaolin as kal
import kornia
import numpy as np
import torch
import torchvision
from PIL import Image

from dos.utils.config import config_to_primitive


def tensor_to_image(tensor, chw=True):
    """
    Convert a tensor to a PIL image
    """
    if len(tensor.shape) == 4:
        if not chw:
            tensor = tensor.permute(0, 3, 1, 2)
            chw = True
        tensor = torchvision.utils.make_grid(
            tensor, nrow=int(math.sqrt(tensor.shape[0]))
        )
    if chw:
        tensor = tensor.permute(1, 2, 0)
    return Image.fromarray((tensor * 255).detach().cpu().numpy().astype(np.uint8))


def dino_features_to_image(
    patch_key, dino_pca_mat, h=256, w=256, dino_feature_recon_dim=3
):
    """
    Convert DINO features to an image
    """
    dino_feat_im = patch_key.reshape(-1, patch_key.shape[-1]).cpu().numpy()
    dims = dino_feat_im.shape[:-1]
    dino_feat_im = dino_feat_im / np.linalg.norm(dino_feat_im, axis=1, keepdims=True)
    dino_feat_im = (
        torch.from_numpy(dino_pca_mat.apply_py(dino_feat_im))
        .to(patch_key.device)
        .reshape(*dims, -1)
    )
    dino_feat_im = (
        dino_feat_im.reshape(-1, 32, 32, dino_feat_im.shape[-1])
        .permute(0, 3, 1, 2)
        .clip(-1, 1)
        * 0.5
        + 0.5
    )
    # TODO: is it needed?
    dino_feat_im = torch.nn.functional.interpolate(
        dino_feat_im, size=[h, w], mode="bilinear"
    )[:, :dino_feature_recon_dim]
    return dino_feat_im


def blender_to_opengl(matrix):
    """
    Convert the camera matrix from Blender to OpenGL coordinate system
    """
    device = matrix.device
    # fmt: off
    conversion_mat = torch.Tensor([
                                [1, 0, 0, 0],
                                [0, 0, 1, 0],
                                [0, -1, 0, 0],
                                [0, 0, 0, 1]]).float().to(device)
    # fmt: on
    return conversion_mat @ matrix


def matrix_to_rotation_translation(matrix):
    """
    matrix is a 4x4 matrix in the OpenGL coordinate system

    retruns rotation and translation, rotation is represented by a quaternion, translation is a 3D vector
    """
    rotation, translation = kornia.geometry.conversions.matrix4x4_to_Rt(
        matrix.contiguous()
    )
    rotation = kornia.geometry.conversions.rotation_matrix_to_quaternion(
        rotation.contiguous()
    )
    return rotation, translation.squeeze(2)


def rotation_translation_to_matrix(rotation, translation):
    """
    rotation is a quaternion, translation is a 3D vector

    returns matrix, which is a 4x4 matrix in the OpenGL coordinate system

    supports batched inputs
    """
    rotation = kornia.geometry.conversions.quaternion_to_rotation_matrix(rotation)
    return kornia.geometry.conversions.Rt_to_matrix4x4(
        rotation, translation.unsqueeze(2)
    )


class MeshRenderer(object):
    def __init__(self, obj_path=None, device="cuda"):
        self.device = device
        if obj_path is not None:
            self.mesh = self.load_obj(obj_path)

    def load_obj(self, obj_path):
        mesh = kal.io.obj.import_mesh(obj_path, with_materials=True)
        vertices = mesh.vertices.to(self.device).unsqueeze(0)
        faces = mesh.faces.to(self.device)
        uvs = mesh.uvs.to(self.device).unsqueeze(0)
        face_uvs_idx = mesh.face_uvs_idx.to(self.device)
        face_uvs = kal.ops.mesh.index_vertices_by_faces(uvs, face_uvs_idx).detach()
        texture_map = (
            mesh.materials[0]["map_Kd"].float().to(self.device).permute(2, 0, 1)[None]
            / 255
        )
        return {
            "vertices": vertices,
            "faces": faces,
            "face_uvs": face_uvs,
            "texture_map": texture_map,
        }

    def render(self, view_matrix, fov=50, obj_path=None):
        """
        Render the mesh using the camera matrix
        view_matrix: 4x4 matrix in the OpenGL coordinate system
        """
        if obj_path is not None:
            mesh = self.load_obj(obj_path)
        else:
            mesh = self.mesh
        vertices, faces, face_uvs, texture_map = (
            mesh[k] for k in ["vertices", "faces", "face_uvs", "texture_map"]
        )

        batch_size = view_matrix.shape[0]

        cam_proj = kal.render.camera.generate_perspective_projection(
            math.radians(fov), ratio=1.0, dtype=torch.float32
        ).to(self.device)

        # opengl is column major, but kaolin is row major
        view_matrix = view_matrix[:, :3].permute(0, 2, 1)

        (
            face_vertices_camera,
            face_vertices_image,
            face_normals,
        ) = kal.render.mesh.prepare_vertices(
            vertices.repeat(batch_size, 1, 1),
            faces,
            cam_proj,
            camera_transform=view_matrix,
        )

        ### Perform Rasterization ###
        # Construct attributes that DIB-R rasterizer will interpolate.
        # the first is the UVS associated to each face
        # the second will make a hard segmentation mask
        nb_faces = faces.shape[0]
        face_attributes = [
            face_uvs.repeat(batch_size, 1, 1, 1),
            torch.ones((batch_size, nb_faces, 3, 1), device=self.device),
        ]

        # If you have nvdiffrast installed you can change rast_backend to
        # nvdiffrast or nvdiffrast_fwd
        image_features, soft_mask, face_idx = kal.render.mesh.dibr_rasterization(
            256,
            256,
            face_vertices_camera[:, :, :, -1],
            face_vertices_image,
            face_attributes,
            face_normals[:, :, -1],
            rast_backend="cuda",
        )

        # image_features is a tuple in composed of the interpolated attributes of face_attributes
        texture_coords, mask = image_features
        # texture_coords = image_features
        # image = image_features
        image = kal.render.mesh.texture_mapping(
            texture_coords, texture_map.repeat(batch_size, 1, 1, 1), mode="bilinear"
        )
        image = torch.clamp(image * mask, 0.0, 1.0)

        return image


@dataclass
class RandomMaskOccluder(object):
    num_occluders_range = (1, 6)
    min_size = 0.1
    max_size = 0.3

    def __call__(self, masks):
        # Get the input tensor shape
        batch_size, _, height, width = masks.shape
        masks = masks.clone()

        # Iterate over images in the batch
        # TODO: vectorize this
        for i in range(batch_size):
            num_occlusions = random.randint(*self.num_occluders_range)
            # Create multiple occlusions per image
            min_size = int(self.min_size * min(height, width))
            max_size = int(self.max_size * min(height, width))
            for _ in range(num_occlusions):
                # Define occlusion size
                occlusion_size_x = random.randint(min_size, max_size)
                occlusion_size_y = random.randint(min_size, max_size)

                # Define occlusion position
                occlusion_x = random.randint(0, width - occlusion_size_x)
                occlusion_y = random.randint(0, height - occlusion_size_y)

                # Create occlusion on all channels
                masks[
                    i,
                    :,
                    occlusion_y : occlusion_y + occlusion_size_y,
                    occlusion_x : occlusion_x + occlusion_size_x,
                ] = 0

        return masks


def rgetattr(obj, attr, *args):
    # https://stackoverflow.com/questions/31174295/getattr-and-setattr-on-nested-subobjects-chained-properties
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)

    return functools.reduce(_getattr, [obj] + attr.split("."))


def safe_batch_to_device(batch, *args, **kwargs):
    out_batch = {}
    for k, v in batch.items():
        if hasattr(v, "to"):
            out_batch[k] = v.to(*args, **kwargs)
        else:
            out_batch[k] = v
    return out_batch


def create_video(dir_path, out_video_name):
    out_path = dir_path
    if not os.path.exists(out_path):
    	os.makedirs(out_path)
    
    out_video_full_path = out_path + out_video_name
    def sort_numeric(file):
        # Extract numbers from the filename and convert them to integers
        return int(''.join(filter(str.isdigit, file)))
    # Ensure images are processed in numerical order
    pre_imgs = sorted(os.listdir(path), key=sort_numeric)
    # print(pre_imgs)
    img = []
    # Limiting the number of images
    for i, item in tqdm(enumerate(pre_imgs)):    # to specify the end image (pre_imgs[:140])
        item = os.path.join(path, item)
        print(item)
        img.append(item)
    # print(img)
    cv2_fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # print(img[0])
    frame = cv2.imread(img[0])
    #print("frame's type", type(frame))
    size = list(frame.shape)
    #print("type", type(size))
    #print('size', size)
    del size[2]
    size.reverse()
    #print("size", size)
    # Reducing fps to make video slower - frames per sec (fps) is 5
    video = cv2.VideoWriter(out_video_full_path, cv2_fourcc, 3, (size[0],size[1]))
    #video = cv2.VideoWriter(out_video_full_path, apiPreference = 0, fourcc = cv2_fourcc, fps = 25, frameSize = size, isColor = True) #output video name, fourcc, fps, size
    for i in range(len(img)): 
        video.write(cv2.imread(img[i]))
        print('frame ', i, ' of ', len(img))
    video.release()
    print('outputed video to ', out_path)
    

def C(value, epoch: int, global_step: int) -> float:
    if isinstance(value, int) or isinstance(value, float):
        pass
    else:
        value = config_to_primitive(value)
        if not isinstance(value, list):
            raise TypeError("Scalar specification only supports list, got", type(value))
        if len(value) == 3:
            value = [0] + value
        assert len(value) == 4
        start_step, start_value, end_value, end_step = value
        if isinstance(end_step, int):
            current_step = global_step
            value = start_value + (end_value - start_value) * max(
                min(1.0, (current_step - start_step) / (end_step - start_step)), 0.0
            )
        elif isinstance(end_step, float):
            current_step = epoch
            value = start_value + (end_value - start_value) * max(
                min(1.0, (current_step - start_step) / (end_step - start_step)), 0.0
            )
    return value
    