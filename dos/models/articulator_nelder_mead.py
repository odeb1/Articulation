import random
import os
# # Set TORCH_HOME to a custom directory
# os.environ['TORCH_HOME'] = '/work/oishideb/cache/torch_hub'
import time
import timeit

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as nn_functional
import torchvision.transforms.functional as F
import torchvision
from PIL import Image, ImageDraw

from dos.components.fuse.compute_correspond import \
    ComputeCorrespond
from dos.utils.correspondence import (draw_correspondences_1_image,
                                      padding_tensor, resize,
                                      tensor_to_matplotlib_figure,
                                      draw_correspondences_combined)

from ..components.diffusion_model_text_to_image.diffusion_sds import \
    DiffusionForTargetImg
from ..components.skinning.bones_estimation import BonesEstimator
from ..components.skinning.skinning import mesh_skinning
from ..modules.renderer import Renderer
from ..nvdiffrec.render.mesh import load_mesh
from ..predictors.articulation_predictor import ArticulationPredictor
from ..predictors.texture import TexturePredictor
from ..utils import geometry as geometry_utils
from ..utils import mesh as mesh_utils
from ..utils import multi_view, utils
from ..utils import visuals as visuals_utils
from .base import BaseModel
from dos.nvdiffrec.render.mesh import make_mesh

from torchvision.transforms import ToPILImage
import clip
from scipy.optimize import minimize
import pandas as pd


class Articulator(BaseModel):
    """
    Articulator predicts instance shape parameters (instance shape) - optimisation based - predictor takes only id as input
    """
    # TODO: set default values for the parameters (dataclasses have a nice way of doing it
    #   but it's not compatible with torch.nn.Module)
    
    def __init__(
        self,
        path_to_save_images,
        num_pose_for_optim,
        num_pose_for_visual,
        num_sample_bone_line,
        num_sample_farthest_points = 100,
        mode_kps_selection = "kps_fr_sample_on_bone_line",
        enable_texture_predictor=True,
        texture_predictor=None,
        bones_predictor=None,
        gltf_skin=None,
        articulation_predictor=None,
        renderer=None,
        shape_template_path=None,
        view_option = "multi_view_azimu",
        fit_shape_template_inside_unit_cube=False,
        use_gt_target_img=False,
        diffusion_Text_to_Target_Img=None,
        sds_every_n_iter=1,
        device = "cuda",
        correspond = None,
        # TODO: Create a view sampler class to encapsulate the view sampling logic and settings
        random_camera_radius= 2.5,  #[2.5, 2.5],
        phi_range_for_optim = [90,90],
        phi_range_for_visual = [0, 360],
        cyc_check_img_save = False,
        bones_rotations = "bones_rotations",
        using_pil_object = False,
        cyc_consi_check_switch = True,
        cyc_consi_check_dist_threshold = 15,
        seed = 50,
        target_image_fixed = False,
        save_individual_img = False,
        multi_view_optimise_option = 'random_phi_each_step_along_azimuth',
        
        # For Nelder Mead Optimisation
        model_name="ViT-L/14@336px", 
        text_prompt="A Cow running very fast", 
        batch_size=1, 
        max_iterations=200, 
        output_dir="intermediate_images_bones_rotations",
    ):
        super().__init__()
        self.path_to_save_images = path_to_save_images
        self.num_pose_for_optim = num_pose_for_optim
        self.num_pose_for_visual = num_pose_for_visual
        self.num_sample_bone_line = num_sample_bone_line
        self.num_sample_farthest_points = num_sample_farthest_points
        self.mode_kps_selection = mode_kps_selection
        self.enable_texture_predictor = enable_texture_predictor
        self.texture_predictor = (
            texture_predictor if texture_predictor is not None else TexturePredictor()
        )
        self.bones_predictor = (
            bones_predictor if bones_predictor is not None else BonesEstimator()
        )
        self.gltf_skin = gltf_skin
        # Articulation predictor
        self.articulation_predictor = (articulation_predictor if articulation_predictor else ArticulationPredictor())
        self.renderer = renderer if renderer is not None else Renderer()
        self.view_option = view_option
        
        if shape_template_path is not None:
            self.shape_template = self._load_shape_template(shape_template_path, fit_inside_unit_cube=fit_shape_template_inside_unit_cube)    # False if self.view_option == "single_view" else True
        else:
            self.shape_template = None

        self.use_gt_target_img = use_gt_target_img
        
        self.diffusion_Text_to_Target_Img = diffusion_Text_to_Target_Img if diffusion_Text_to_Target_Img is not None else DiffusionForTargetImg()
        self.sds_every_n_iter = sds_every_n_iter
        self.device = device
        self.correspond = (correspond if correspond else ComputeCorrespond())
        self.random_camera_radius = random_camera_radius    # 1 if self.view_option == "single_view" else 2.5
        self.phi_range_for_optim = phi_range_for_optim
        self.phi_range_for_visual = phi_range_for_visual
        self.cyc_check_img_save = cyc_check_img_save
        self.bones_rotations = bones_rotations
        self.using_pil_object = using_pil_object
        self.cyc_consi_check_switch = cyc_consi_check_switch
        self.cyc_consi_check_dist_threshold = cyc_consi_check_dist_threshold
        self.seed = random.seed(seed)
        self.target_image_fixed = target_image_fixed
        self.save_individual_img = save_individual_img
        self.multi_view_optimise_option = multi_view_optimise_option
        
        # For Nelder Mead Optimisation
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        self.text_prompt = text_prompt
        self.text = clip.tokenize([self.text_prompt]).to(self.device)
        self.bones_rotations_shape = (batch_size, 47, 4)
        self.similarity_scores = []
        self.output_dir = output_dir
        self.max_iterations = max_iterations
        self.iteration = 0
        self.to_pil = ToPILImage()  # Added transform to convert tensor to PIL Image
        self.gradient_free_optim = True
        

    def _load_shape_template(self, shape_template_path, fit_inside_unit_cube=False):
        mesh = load_mesh(shape_template_path)
        # position the mesh inside the unit cube
        if fit_inside_unit_cube:
            mesh = mesh_utils.fit_inside_unit_cube(mesh)
        return mesh

    
    def forward(self, batch, num_batches, iteration):
        
        batch_size = batch["image"].shape[0]
        if self.shape_template is not None:
            mesh = self.shape_template.extend(batch_size)
        else:
            mesh = batch["mesh"]  # rest pose
        
        # estimate bones
        # bones_predictor_outputs is dictionary with keys - ['bones_pred', 'skinnig_weights', 'kinematic_chain', 'aux'])
        start_time = time.time()
        bones_predictor_outputs = self.bones_predictor(mesh.v_pos)
        end_time = time.time()  # Record the end time
        # with open('log.txt', 'a') as file:
        #     file.write(f"The 'bones_predictor' took {end_time - start_time} seconds to run.\n")
                
        print(f"The bones_predictor function took {end_time - start_time} seconds to run.")

        batch_size, _ = bones_predictor_outputs["bones_pred"].shape[:2]
        
        if self.bones_rotations == "bones_rotations":
            start_time = time.time()
            # BONE ROTATIONS
            # bones_rotations is of shape [batch size, 47, 4]
            bones_rotations = self.articulation_predictor(batch)
            end_time = time.time()  # Record the end time
            # with open('log.txt', 'a') as file:
            #     file.write(f"The 'articulation_predictor' took {end_time - start_time} seconds to run.\n")
            print(f"The articulation_predictor function took {end_time - start_time} seconds to run.")
            
        elif self.bones_rotations == "NO_bones_rotations":
            # NO BONE ROTATIONS
            bones_rotations = torch.zeros(batch_size, 47, 4, device=mesh.v_pos.device)
        
        elif self.bones_rotations == "DUMMY_bones_rotations":
            # DUMMY BONE ROTATIONS - pertrub the bones rotations (only to test the implementation)
            bones_rotations = bones_rotations + torch.randn_like(bones_rotations) * 0.1
    
    
        start_time = time.time()  # Record the start time
        # apply articulation to mesh
        if self.gltf_skin is not None:
            articulated_verts, skin_aux = self.gltf_skin.skin_mesh_with_rotations(mesh.v_pos, bones_rotations)
            articulated_mesh = make_mesh(
                articulated_verts, mesh.t_pos_idx, mesh.v_tex, mesh.t_tex_idx, mesh.material
            )
            posed_bones = None
        else:
            articulated_mesh, skin_aux = mesh_skinning(
                mesh,
                bones_predictor_outputs["bones_pred"],
                bones_predictor_outputs["kinematic_chain"],
                bones_rotations,
                bones_predictor_outputs["skinnig_weights"],
                output_posed_bones=True,
            )
            posed_bones = skin_aux["posed_bones"]
            
        end_time = time.time()  # Record the end time
        # with open('log.txt', 'a') as file:
        #     file.write(f"The 'mesh_skinning' took {end_time - start_time} seconds to run.\n")
        print(f"The mesh_skinning function took {end_time - start_time} seconds to run.")
        
        #articulated_bones_predictor_outputs = self.bones_predictor(articulated_mesh.v_pos)
        
        # render mesh
        if "texture_features" in batch:
            texture_features = batch["texture_features"]
        else:
            texture_features = None

        if self.enable_texture_predictor:
            start_time = time.time()
            material = self.texture_predictor
            end_time = time.time()
            print(f"The function took {end_time - start_time} seconds to run.\n")
        else:
            # if texture predictor is not enabled, use the loaded material from the mesh
            material = mesh.material

        # if pose not provided, compute it from the camera matrix
        if "pose" not in batch:
            if self.view_option == "single_view":
                # For Single View
                # pose shape is [1,12]
                pose = geometry_utils.blender_camera_matrix_to_magicpony_pose(
                    batch["camera_matrix"]
                )
            elif self.view_option == "multi_view_rand":
                # For Multi View
                # pose shape is [num_pose, 12]
                pose, direction = multi_view.rand_poses(self.num_pose_for_optim, self.device, iteration=iteration, radius_range=self.random_camera_radius)
                
            elif self.view_option == "multi_view_azimu":
               
                pose, direction = multi_view.poses_along_azimuth(self.num_pose_for_optim, self.device, batch_number=num_batches, iteration=iteration, radius=self.random_camera_radius, phi_range=self.phi_range_for_optim, multi_view_option = self.multi_view_optimise_option)
        else:
            pose=batch["pose"]
        
        if "background" in batch:
            background = batch["background"]
        else:
            background = None

        start_time = time.time()
        
        renderer_outputs = self.renderer(
            articulated_mesh,
            material=material,
            pose=pose,
            im_features=texture_features
            # CHANGED IT
            # background=background,
        )
        end_time = time.time()  # Record the end time
        # with open('log.txt', 'a') as file:
        #     file.write(f"The 'renderer' took {end_time - start_time} seconds to run.\n")
        print(f"The renderer function took {end_time - start_time} seconds to run.")
        
        start_time = time.time()
        # get visible vertices
        mvp, w2c, _ = geometry_utils.get_camera_extrinsics_and_mvp_from_pose(
            pose,
            self.renderer.fov,
            self.renderer.znear,
            self.renderer.zfar,
            self.renderer.cam_pos_z_offset,
        )
        end_time = time.time()
        # with open('log.txt', 'a') as file:
        #     file.write(f"The 'get_camera_extrinsics_and_mvp_from_pose' compute took {end_time - start_time} seconds to run.\n")
        print(f"The 'get_camera_extrinsics_and_mvp_from_pose' function took {end_time - start_time} seconds to run.")
        
        # For Debugging purpose, save all the poses before optimisation
        self.save_all_poses_before_optimisation(pose, renderer_outputs, self.path_to_save_images)
        
        rendered_image = renderer_outputs["image_pred"]
        
        # Gradient free optimisation
        if self.gradient_free_optim:
            
            optimized_bones_rotations = self.run_optimization(bones_rotations)
            optimized_bones_rotations = torch.tensor(optimized_bones_rotations)
            
            if self.gltf_skin is not None:
                articulated_verts, skin_aux = self.gltf_skin.skin_mesh_with_rotations(mesh.v_pos, optimized_bones_rotations)
                articulated_mesh = make_mesh(
                    articulated_verts, mesh.t_pos_idx, mesh.v_tex, mesh.t_tex_idx, mesh.material
                )
                posed_bones = None
                
            renderer_outputs = self.renderer(
            articulated_mesh,
            material=material,
            pose=pose,
            im_features=texture_features
            )
            
            rendered_image = renderer_outputs["image_pred"]
            
            for i in range(rendered_image.shape[0]):
                rendered_image_PIL = F.to_pil_image(rendered_image[0])
                dir_path = f'{self.path_to_save_images}/nelder_mead_optim/'
                # Create the directory if it doesn't exist
                os.makedirs(dir_path, exist_ok=True)
                # Save the image
                rendered_image_PIL.save(f'{dir_path}{i}_nelder_mead_optim_image.png', bbox_inches='tight')
                
            outputs = {}
            outputs.update(renderer_outputs)

            ## Saving poses along the azimuth
            self.save_pose_along_azimuth(articulated_mesh, material, self.path_to_save_images)      
        
        return outputs
    
    
    def gradient_free_optim_eval(self, bones_rotations):
        # **Apply bones_rotations to the mesh and render the image within this function**
        bones_rotations_tensor = torch.tensor(bones_rotations, dtype=torch.float32).to(self.device)
        bones_rotations_tensor = bones_rotations_tensor.view(self.bones_rotations_shape)  # Reshape to [batch_size, 47, 4]
        
        # **Re-apply articulations and rendering with updated bones_rotations**
        if self.gltf_skin is not None:
            articulated_verts, _ = self.gltf_skin.skin_mesh_with_rotations(self.shape_template.v_pos, bones_rotations_tensor)
            articulated_mesh = make_mesh(
                articulated_verts, self.shape_template.t_pos_idx, self.shape_template.v_tex, self.shape_template.t_tex_idx, self.shape_template.material
            )

        # **Render the articulated mesh**
        renderer_outputs = self.renderer(
            articulated_mesh,
            material=self.shape_template.material,
            pose=self.pose,   #self.current_pose,
            im_features=None
        )
        rendered_image = renderer_outputs["image_pred"]

        # **Preprocess rendered_image for CLIP**
        rendered_image_resized = F.resize(rendered_image, self.clip_input_resolution)  # Ensure correct input size
        to_pil_image = ToPILImage()  # Create a ToPILImage transform
        rendered_image_pil = to_pil_image(rendered_image_resized.squeeze(0).cpu())  # Convert tensor to PIL Image
    
        # **Preprocess rendered_image for CLIP**
        image = self.preprocess(rendered_image_pil).unsqueeze(0).to(self.device)
        
        # Compute CLIP features and similarity score
        with torch.no_grad():
            image_features = self.model.encode_image(image)
            text_features = self.model.encode_text(self.text)

        # Normalize features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # Compute similarity score between image and text prompt
        similarity = torch.matmul(image_features, text_features.T).item()

        # print(f"Similarity: {similarity}")

        return -similarity  # Return negative similarity for minimization


    # Added for Nelder-Mead Optimisation
    def callbackF(self, Xi):
        score = -self.gradient_free_optim_eval(Xi)
        self.similarity_scores.append(score)
        print(f"Iteration {self.iteration}: Similarity Score = {score}")

        # Save intermediate rendered images for visualization**
        self.save_intermediate_images = True
        if self.save_intermediate_images:
            bones_rotations_tensor = torch.tensor(Xi, dtype=torch.float32).to(self.device)
            # **Re-render image with current bones_rotations**
            # rendered_image = self.render_image(bones_rotations_tensor)
            # rendered_image_pil = self.to_pil(rendered_image.squeeze(0).cpu())
            # rendered_image_pil.save(os.path.join(self.output_dir, f"intermediate_image_{self.iteration}.png"))

        self.iteration += 1


    # Added for Nelder-Mead Optimisation
    def run_optimization(self, initial_bones_rotations):
        print("Starting optimization...")
        start_time = time.time()

        # self.shape_template = self.shape_template.to(self.device)
        self.clip_input_resolution = self.preprocess.transforms[0].size  # **CLIP input resolution**
        self.iteration = 0
        self.pose, _ = multi_view.poses_along_azimuth(self.num_pose_for_optim, self.device, batch_number=1, iteration=self.iteration, radius=self.random_camera_radius, phi_range=self.phi_range_for_optim, multi_view_option = self.multi_view_optimise_option)

        # **Prepare initial bones_rotations**
        if isinstance(initial_bones_rotations, torch.Tensor):
            initial_bones_rotations = initial_bones_rotations.detach().cpu().numpy()

        result = minimize(
            fun=self.gradient_free_optim_eval,
            x0=initial_bones_rotations,
            method='Nelder-Mead',
            callback=self.callbackF,
            options={'maxiter': self.max_iterations, 'disp': True}
        )

        end_time = time.time()
        elapsed_time = end_time - start_time

        optimized_bones_rotations = result.x.reshape(self.bones_rotations_shape).astype(np.float32)

        print(f"Optimization completed. Elapsed time: {elapsed_time:.2f} seconds")

        # **Save results**
        self.save_similarity_scores()
        self.plot_minimization_curve()

        return optimized_bones_rotations


    # Added for Nelder-Mead Optimisation
    def save_similarity_scores(self):
        scores_df = pd.DataFrame(self.similarity_scores, columns=["Similarity Score"])
        scores_df.to_csv("similarity_scores.csv", index=False)
        print("Similarity scores saved to 'similarity_scores.csv'")


    # Added for Nelder-Mead Optimisation
    def plot_minimization_curve(self):
        plt.plot(self.similarity_scores)
        plt.xlabel("Iteration")
        plt.ylabel("Similarity Score")
        plt.title("Minimization Curve")
        plt.savefig("minimization_curve.png")
        plt.show()
        print("Minimization curve saved as 'cow_walking_minimization_curve.png'")
    
    
    def get_metrics_dict(self, model_outputs, batch):
        return {}

    def get_loss_dict(self, model_outputs, batch, metrics_dict):
        
        # Keypoint loss
        # Computes the loss between the source and target keypoints
        print('Calculating l2 loss')
        # loss = nn_functional.mse_loss(rendered_keypoints, target_keypoints, reduction='mean')
        model_outputs["rendered_kps"] = model_outputs["rendered_kps"].to(self.device)
        model_outputs["target_corres_kps"] = model_outputs["target_corres_kps"].to(self.device)

        loss = nn_functional.mse_loss(model_outputs["rendered_kps"], model_outputs["target_corres_kps"], reduction='mean')
       
        return {"loss": loss}

    def get_visuals_dict(self, model_outputs, batch, num_visuals=1):
        def _get_visuals_dict(input_dict, names):
            return visuals_utils.get_visuals_dict(input_dict, names, num_visuals)

        visuals_dict = {}

        batch_visuals_names = ["image"]
        visuals_dict.update(_get_visuals_dict(batch, batch_visuals_names))

        model_outputs_visuals_names = ["image_pred"]
        visuals_dict.update(
            _get_visuals_dict(model_outputs, model_outputs_visuals_names)
        )

        # TODO: render also rest pose

        # Log skinned mesh
        if self.gltf_skin is not None:
            visuals_dict["skinned_mesh"] = self.gltf_skin.plot_skinned_mesh_3d(
                model_outputs["articulated_mesh"].v_pos[0].detach().cpu(), 
                model_outputs["skin_aux"]["global_joint_transforms"][0].detach().cpu())

        return visuals_dict
    
    ## Saving all poses
    def save_multiple_random_poses(self, model_outputs, path_to_save_images):
        
        for index, item in enumerate(model_outputs["rendered_image_with_kps"]):
        
            if self.save_individual_img:
                # With KPs visualisation - these images are Matplotlib Object
                if not os.path.exists(f'{path_to_save_images}/all_poses_rendered_img_with_KPs'):
                    os.makedirs(f'{path_to_save_images}/all_poses_rendered_img_with_KPs')
                plt.gcf().set_facecolor('grey')
                model_outputs["rendered_image_with_kps"][index].savefig(f'{path_to_save_images}/all_poses_rendered_img_with_KPs/{index}_poses_rendered_img_with_KPs.png', bbox_inches='tight')
                if not os.path.exists(f'{path_to_save_images}/all_poses_target_img_with_KPs'):
                    os.makedirs(f'{path_to_save_images}/all_poses_target_img_with_KPs')
                plt.gcf().set_facecolor('grey')
                model_outputs["target_image_with_kps"][index].savefig(f'{path_to_save_images}/all_poses_target_img_with_KPs/{index}_poses_target_img_with_KPs.png', bbox_inches='tight')
                
                # Without KPs visualisation - these images are PIL Object
                if not os.path.exists(f'{path_to_save_images}/all_poses_rendered_img_NO_KPs'):
                    os.makedirs(f'{path_to_save_images}/all_poses_rendered_img_NO_KPs')
                model_outputs["rendered_img_NO_kps"][index].save(f'{path_to_save_images}/all_poses_rendered_img_NO_KPs/{index}_poses_rendered_img_NO_KPs.png', bbox_inches='tight')
                
                if not os.path.exists(f'{path_to_save_images}/all_poses_target_img_NO_KPs'):
                    os.makedirs(f'{path_to_save_images}/all_poses_target_img_NO_KPs')
                model_outputs["target_img_NO_kps"][index].save(f'{path_to_save_images}/all_poses_target_img_NO_KPs/{index}_poses_target_img_NO_KPs.png', bbox_inches='tight')
                
            if not os.path.exists(f'{path_to_save_images}/all_poses_rendered_target_combined'):
                os.makedirs(f'{path_to_save_images}/all_poses_rendered_target_combined')
            model_outputs["rendered_target_image_with_wo_kps_list"][index].save(f'{path_to_save_images}/all_poses_rendered_target_combined/{index}_all_poses_rendered_target_combined.png', bbox_inches='tight')
            

    ## Saving poses along the azimuth for Visualisation
    def save_pose_along_azimuth(self, articulated_mesh, material, path_to_save_images):
        
        if self.view_option == "single_view":
            # Added for debugging purpose
            pose, direction = multi_view.poses_along_azimuth_single_view(self.num_pose_for_visual, device=self.device)
        else:
            pose, direction = multi_view.poses_along_azimuth(self.num_pose_for_visual, device=self.device, radius=self.random_camera_radius, phi_range=self.phi_range_for_visual, multi_view_option ='multiple_random_phi_in_batch')
        
        renderer_outputs = self.renderer(
            articulated_mesh,
            material= material,
            pose=pose,
            im_features= None
        )
        
        for i in range(pose.shape[0]):
            rendered_image_PIL = F.to_pil_image(renderer_outputs["image_pred"][i])
            rendered_image_PIL = resize(rendered_image_PIL, target_res = 840, resize=True, to_pil=True)
            dir_path = f'{path_to_save_images}/azimuth_pose/rendered_img/'
            # Create the directory if it doesn't exist
            os.makedirs(dir_path, exist_ok=True)
            # Save the image
            rendered_image_PIL.save(f'{dir_path}_azimuth_pose_rendered_image.png', bbox_inches='tight')
            
            
    # Saving Rendered Image at every iteration with keypoints visualisation
    def save_img_each_iteration(self, model_outputs, iteration, index_of_image, path_to_save_img_per_iteration):
        
        start_time = time.time()
        
        for index, item in enumerate(model_outputs["rendered_image_with_kps"]):
            
            if self.save_individual_img:  
                # This image is a Matplotlib Object
                dir_path = f'{path_to_save_img_per_iteration}/rendered_img_with_kps/{index}_pose'
                os.makedirs(dir_path, exist_ok=True)
                plt.gcf().set_facecolor('grey')
                model_outputs["rendered_image_with_kps"][index].savefig(f'{dir_path}/{iteration}_rendered_img_with_kps.png', bbox_inches='tight')

                # This image is a PIL Object
                dir_path = f'{path_to_save_img_per_iteration}/rendered_img_NO_kps/{index}_pose'
                os.makedirs(dir_path, exist_ok=True)
                model_outputs["rendered_img_NO_kps"][index].save(f'{dir_path}/{iteration}_rendered_img_NO_kps.png', bbox_inches='tight')

                # This image is a Matplotlib Object
                plt.gcf().set_facecolor('grey')
                dir_path = f'{path_to_save_img_per_iteration}/target_img_with_kps/{index}_pose'
                os.makedirs(dir_path, exist_ok=True)
                model_outputs["target_image_with_kps"][index].savefig(f'{dir_path}/{iteration}_target_img_with_kps.png', bbox_inches='tight')
                
                # This image is a PIL Object
                dir_path = f'{path_to_save_img_per_iteration}/target_img_NO_kps/{index}_pose'
                os.makedirs(dir_path, exist_ok=True)
                model_outputs["target_img_NO_kps"][index].save(f'{dir_path}/{iteration}_target_img_NO_kps.png', bbox_inches='tight')
                
            dir_path = f'{path_to_save_img_per_iteration}/rendered_target_image_with_wo_kps_list/{index}_pose'
            os.makedirs(dir_path, exist_ok=True)
            model_outputs["rendered_target_image_with_wo_kps_list"][index].save(f'{dir_path}/{iteration}_rendered_target_image_with_wo_kps_list.png', bbox_inches='tight')
            
            
            if (self.cyc_consi_check_switch & self.cyc_check_img_save):
                dir_path = f'{path_to_save_img_per_iteration}/cyc_check_combined_image_list/{index}_pose'
                os.makedirs(dir_path, exist_ok=True)
                model_outputs["cyc_check_combined_image_list"][index].save(f'{dir_path}/{iteration}_cyc_check_combined_image_list.png', bbox_inches='tight')
             
        
            end_time = time.time()  # Record the end time
            print(f"The 'Saving img for every iterations' took {end_time - start_time} seconds to run.")
            # with open('log.txt', 'a') as file:
            #     file.write(f"The 'Saving img for every iterations' took {end_time - start_time} seconds to run.\n")
    
    
    # For Debugging purpose, save all the poses before optimisation
    def save_all_poses_before_optimisation(self, pose, renderer_outputs, path_to_save_images):    
        for i in range(pose.shape[0]):
            rendered_image_PIL = F.to_pil_image(renderer_outputs["image_pred"][i])
            #rendered_image_PIL = resize(rendered_image_PIL, target_res = 840, resize=True, to_pil=True)
            dir_path = f'{self.path_to_save_images}/diff_pose/rendered_img/'
            # Create the directory if it doesn't exist
            os.makedirs(dir_path, exist_ok=True)
            # Save the image
            rendered_image_PIL.save(f'{dir_path}{i}_diff_pose_rendered_image.png', bbox_inches='tight')
