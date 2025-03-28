import random
import numpy as np
import torch

# ADDED FOR MULTI-VIEW/3D
def safe_normalize(x, eps=1e-20):
    return x / torch.sqrt(torch.clamp(torch.sum(x * x, -1, keepdim=True), min=eps))

def get_view_direction(thetas, phis, overhead, front, phi_offset=0):
    #                   phis [B,];          thetas: [B,]
    # front = 0         [360 - front / 2, front / 2)
    # side (left) = 1   [front / 2, 180 - front / 2)
    # back = 2          [180 - front / 2, 180 + front / 2)
    # side (right) = 3  [180 + front / 2, 360 - front / 2)
    # top = 4                               [0, overhead]
    # bottom = 5                            [180-overhead, 180]
    res = torch.zeros(thetas.shape[0], dtype=torch.long)

    # first determine by phis
    phi_offset = np.deg2rad(phi_offset)
    phis = phis + phi_offset
    phis = phis % (2 * np.pi)
    half_front = front / 2
    
    res[(phis >= (2*np.pi - half_front)) | (phis < half_front)] = 0
    res[(phis >= half_front) & (phis < (np.pi - half_front))] = 1
    res[(phis >= (np.pi - half_front)) & (phis < (np.pi + half_front))] = 2
    res[(phis >= (np.pi + half_front)) & (phis < (2*np.pi - half_front))] = 3

    # override by thetas
    res[thetas <= overhead] = 4
    res[thetas >= (np.pi - overhead)] = 5
    return res

def view_direction_id_to_text(view_direction_id):
    dir_texts = ['front', 'side', 'back', 'side', 'overhead', 'bottom']
    return [dir_texts[i] for i in view_direction_id]

def get_2_alternating_phi(device, last_phi=[None]):
    # Use a mutable default argument as a static variable to store the last state
    if last_phi[0] is None or last_phi[0] == np.deg2rad(-90):
        last_phi[0] = np.deg2rad(90)
    else:
        last_phi[0] = np.deg2rad(-90)
    return torch.tensor([last_phi[0]], dtype=torch.float, device=device)

def get_4_alternating_phi(device, last_phi=[-1]):
    # the sequence of phi values to cycle through
    phi_sequence = [np.deg2rad(90), np.deg2rad(180), np.deg2rad(-90), np.deg2rad(360)]
    
    # Use a static variable (last_phi) to store the index of the last phi value in the sequence
    # Increment the index to move to the next value in the sequence
    last_phi[0] = (last_phi[0] + 1) % len(phi_sequence)
    
    # Select the current phi value based on the updated index
    current_phi = phi_sequence[last_phi[0]]
    
    # Return a tensor containing the current phi value
    return torch.tensor([current_phi], dtype=torch.float, device=device)


def get_2_alternating_phi_45_degree_apart(device, last_phi=[-1, 0], update_interval=20):
    """
    Generates phi values that are 45 degrees apart, maintaining each phi value for 'n' iterations.
    
    Args:
        device (str): The device type (e.g., 'cpu' or 'cuda').
        last_phi (list): A list where the first element is the index of the last phi value,
                         and the second element is the iteration count for the current phi.
        n (int): The number of iterations to keep the same phi value.
    
    Returns:
        torch.Tensor: A tensor containing the current phi value.
    """
    # The sequence of phi values to cycle through
    # phi_sequence = [np.deg2rad(45), np.deg2rad(315)]
    
    # For the new target images
    phi_sequence = [np.deg2rad(315), np.deg2rad(45)]
    
    # Check if the current phi value has been used for 'n' iterations
    if last_phi[1] >= update_interval:
        # Increment the index to move to the next value in the sequence
        last_phi[0] = (last_phi[0] + 1) % len(phi_sequence)
        last_phi[1] = 0  # Reset the iteration counter for the new phi value
    else:
        # Increment the iteration counter for the current phi value
        last_phi[1] += 1
    
    # Select the current phi value based on the updated index
    current_phi = phi_sequence[last_phi[0]]
    
    print("phis", torch.tensor([current_phi], dtype=torch.float, device=device))
    
    # Return a tensor containing the current phi value
    return torch.tensor([current_phi], dtype=torch.float, device=device)


def get_4_alternating_phi_45_degree_apart_old(device, update_interval, last_phi=[-1, 0]):
    """
    Generates phi values that are 45 degrees apart, maintaining each phi value for 'n' iterations.
    
    Args:
        device (str): The device type (e.g., 'cpu' or 'cuda').
        last_phi (list): A list where the first element is the index of the last phi value,
                         and the second element is the iteration count for the current phi.
        n (int): The number of iterations to keep the same phi value.
    
    Returns:
        torch.Tensor: A tensor containing the current phi value.
    """
    # The sequence of phi values to cycle through
    #phi_sequence = [np.deg2rad(-45), np.deg2rad(135), np.deg2rad(45), np.deg2rad(-135)]
    # phi_sequence = [np.deg2rad(315), np.deg2rad(45), np.deg2rad(-315), np.deg2rad(-45)]
    phi_sequence = [np.deg2rad(45), np.deg2rad(315), np.deg2rad(-45), np.deg2rad(-315)]
    
    # Check if the current phi value has been used for 'n' iterations
    if last_phi[1] >= update_interval:
        # Increment the index to move to the next value in the sequence
        last_phi[0] = (last_phi[0] + 1) % len(phi_sequence)
        last_phi[1] = 0  # Reset the iteration counter for the new phi value
        print("last_phi[0]", last_phi[0])
    else:
        # Increment the iteration counter for the current phi value
        print("last_phi[1]", last_phi[1])
        last_phi[1] += 1
        print("last_phi[1]", last_phi[1])
    
    # Select the current phi value based on the updated index
    current_phi = phi_sequence[last_phi[0]]
    
    print("phis", torch.tensor([current_phi], dtype=torch.float, device=device))
    
    # Return a tensor containing the current phi value
    return torch.tensor([current_phi], dtype=torch.float, device=device)


def get_4_alternating_phi_45_degree_apart(device, iteration, update_interval=20):
    """
    Generates phi values that are 45 degrees apart, maintaining each phi value for 'update_interval' iterations.

    Args:
        device (str): The device type (e.g., 'cpu' or 'cuda').
        iteration (int): The current iteration number.
        update_interval (int): The number of iterations to keep the same phi value.

    Returns:
        torch.Tensor: A tensor containing the current phi value.
    """
    # Define the sequence of phi values
    phi_sequence = [np.deg2rad(45), np.deg2rad(315), np.deg2rad(225), np.deg2rad(135)]
    
    # Calculate the index in the phi sequence based on the iteration number
    sequence_index = (iteration // update_interval) % len(phi_sequence)
    
    # Select the current phi value from the sequence
    current_phi = phi_sequence[sequence_index]
    
    print(f"Iteration {iteration}: Using phi = {np.rad2deg(current_phi)} degrees")

    # Return the current phi value as a tensor
    return torch.tensor([current_phi], dtype=torch.float, device=device)


# Generates camera poses in 3D space with adjustable parameters for spherical angles, jitter, and translation offsets (i.e cam_z_offset along z-axis).
def poses_helper_func(size, device, phis, thetas, radius_range=[2.5, 2.5], angle_overhead=30, angle_front=60, phi_offset=0, jitter=False, cam_z_offset=0, return_dirs=True):

    # Convert overhead angle from degrees to radians
    angle_overhead = np.deg2rad(angle_overhead)
    
    # Convert front angle from degrees to radians
    angle_front = np.deg2rad(angle_front)
    
    # Generate a random radius for each pose within the specified radius range
    # This creates a range of distances from the target for the camera positions
    radius = torch.rand(size, device=device) * (radius_range[1] - radius_range[0]) + radius_range[0]

    # Initialize target points (center of attention for the camera) at origin (0, 0, 0)
    targets = torch.zeros(size, 3, device=device)
    
    # Calculate camera center positions based on spherical coordinates
    # `phis` and `thetas` determine the azimuth and elevation, respectively
    centers = -torch.stack([
        torch.sin(thetas) * torch.sin(phis),  # x-component
        torch.cos(thetas),                    # y-component
        torch.sin(thetas) * torch.cos(phis),  # z-component
    ], dim=-1)  # Resulting tensor shape is [size, 3]
    
    # Apply random jitter if `jitter` is True, adding small noise to camera positions and targets
    if jitter:
        centers = centers + (torch.rand_like(centers) * 0.2 - 0.1)  # Random variation for centers
        targets = targets + torch.randn_like(centers) * 0.2         # Random variation for targets
    
    # Calculate forward direction vector as the normalized direction from center to target
    forward_vector = safe_normalize(targets - centers)
    
    # Define the up vector (y-axis up direction) and repeat for each pose
    up_vector = torch.FloatTensor([0, 1, 0]).to(device).unsqueeze(0).repeat(size, 1)
    
    # Compute right vector as the cross product of up and forward vectors to ensure orthogonality
    right_vector = safe_normalize(torch.cross(up_vector, forward_vector, dim=-1))
    
    # Optionally add small random noise to the up vector if `jitter` is enabled
    if jitter:
        up_noise = torch.randn_like(up_vector) * 0.02  # Small noise addition for variability
    else:
        up_noise = 0  # No noise if jitter is disabled
    
    # Recalculate up vector to ensure orthogonality and add noise if needed
    up_vector = safe_normalize(torch.cross(forward_vector, right_vector, dim=-1) + up_noise)
    
    # Combine right, up, and forward vectors to form a rotation matrix for each pose
    # The rotation matrix determines the camera's orientation in 3D space
    poses = torch.stack([right_vector, up_vector, forward_vector], dim=-1)
    
    # Adjust radius to account for any offset along the z-axis of the camera
    radius = radius[..., None] - cam_z_offset
    
    # Define translations for each pose based on radius (only along the z-axis)
    # Translations indicate the camera's position in space relative to the target
    translations = torch.cat([torch.zeros_like(radius), torch.zeros_like(radius), -radius], dim=-1) 
    
    # Combine rotation matrix and translation vector to form a full pose vector for each pose
    poses = torch.cat([poses.view(-1, 9), translations], dim=-1)
    
    if return_dirs:
        # for mv_dream option -> phis shape needs to be [4]
        # Obtain view directions based on spherical angles, offsets, and overhead/front angles
        dirs = get_view_direction(thetas, phis, angle_overhead, angle_front, phi_offset=phi_offset)
        # Convert direction IDs to text format
        dirs = view_direction_id_to_text(dirs)
    else:
        dirs = None
    
    return poses, dirs

# # Global variable to store the state
# phis_state = {
#     "last_phis": None,
#     "last_update_iteration": -1,
#     "long_update_interval": 15,  # Interval for side views
#     "short_update_interval": 2,  # Interval for all other views
#     "update_interval": 10
# }

def initialize_phis_state(device):
    """ Initialize the phis state """
    global phis_state
    phis_state["last_phis"] = torch.tensor([np.deg2rad(45)], dtype=torch.float, device=device)
    phis_state["last_update_iteration"] = 0  # Assume starting at iteration 0

    
def is_side_view(phis, tolerance=np.deg2rad(1), device='cpu'):
    ninety_degrees = torch.tensor(np.deg2rad(90), dtype=torch.float, device=device)
    minus_ninety_degrees = torch.tensor(np.deg2rad(-90), dtype=torch.float, device=device)
    return torch.any(torch.isclose(phis, ninety_degrees, atol=tolerance)) or torch.any(torch.isclose(phis, minus_ninety_degrees, atol=tolerance))


def poses_along_azimuth(size, device, update_interval, batch_number=0, iteration=0,  radius=2.5, theta=90, phi_range=[0, 360], multi_view_option='4_side_views_only_in_batch', **kwargs):
    ''' generate random poses from an orbit camera along uniformly distributed azimuth and fixed elevation
    Args:
        size: batch size of generated poses.
        device: where to allocate the output.
        theta: is a constant                          
        phi_range: [min, max], should be in [0, 2 * pi]
    Return:
        poses: [size, 4, 4]
    '''
    global phis_state  # Reference the global variable
    theta = np.deg2rad(theta)
    
    if multi_view_option == '2_side_views_only_in_batch':
        # Side view 1 at 45 degrees and side view 2 at 135 degrees
        phis = torch.tensor([np.deg2rad(45), np.deg2rad(315)], dtype=torch.float, device=device)
        
    elif multi_view_option == '2_side_views_only_in_batch_90_270':
        # Side view 1 at 45 degrees and side view 2 at 135 degrees
        phis = torch.tensor([np.deg2rad(90), np.deg2rad(270)], dtype=torch.float, device=device)
        
    elif multi_view_option == '4_side_views_only_in_batch':
        phis = torch.tensor([np.deg2rad(45), np.deg2rad(315), np.deg2rad(225), np.deg2rad(135)], dtype=torch.float, device=device)    
        
    elif multi_view_option == '4_side_views_only_in_batch_90_360_270_180':
        phis = torch.tensor([np.deg2rad(90), np.deg2rad(360), np.deg2rad(270), np.deg2rad(180)], dtype=torch.float, device=device)    
    
    elif multi_view_option == '4_side_views_only_in_batch_90_135_minus':
        phis = torch.tensor([np.deg2rad(-90), np.deg2rad(-135), np.deg2rad(135), np.deg2rad(90)], dtype=torch.float, device=device)    
   
    elif multi_view_option == '6_side_views_only_in_batch':
        phis = torch.tensor([np.deg2rad(45), np.deg2rad(15), np.deg2rad(315), np.deg2rad(270), np.deg2rad(135), np.deg2rad(90)], dtype=torch.float, device=device)    
   
    elif multi_view_option == '7_side_views_only_in_batch':
        phis = torch.tensor([np.deg2rad(15), np.deg2rad(360), np.deg2rad(270), np.deg2rad(225), np.deg2rad(180), np.deg2rad(135), np.deg2rad(90)], dtype=torch.float, device=device)    
   
    elif multi_view_option == '8_side_views_only_in_batch':
        phis = torch.tensor([np.deg2rad(45), np.deg2rad(360), np.deg2rad(315), np.deg2rad(270), np.deg2rad(225), np.deg2rad(180), np.deg2rad(135), np.deg2rad(90)], dtype=torch.float, device=device)    
    
    elif multi_view_option == '12_side_views_only_in_batch':
        phis = torch.tensor([np.deg2rad(30), np.deg2rad(360), np.deg2rad(330), np.deg2rad(300), np.deg2rad(270), np.deg2rad(240), np.deg2rad(210), np.deg2rad(180), np.deg2rad(150), np.deg2rad(120), np.deg2rad(90), np.deg2rad(60)], dtype=torch.float, device=device)    
   
    elif multi_view_option == '12_side_views_only_in_batch_tiger':
        phis = torch.tensor([np.deg2rad(360), np.deg2rad(340), np.deg2rad(280), np.deg2rad(275), np.deg2rad(270), np.deg2rad(175), np.deg2rad(165), np.deg2rad(135), np.deg2rad(125), np.deg2rad(100), np.deg2rad(95), np.deg2rad(10)], dtype=torch.float, device=device)    
   
    elif multi_view_option == 'guidance_and_rand_views_in_batch':
        # phis selects two values, a fixed side view (45 degrees) and a random view each iteration.  # Here, 1 represents Size i.e 1 random view
        phis = torch.tensor([np.deg2rad(45), torch.rand(1, device=device) * (np.deg2rad(phi_range[1]) - np.deg2rad(phi_range[0])) + np.deg2rad(phi_range[0])
                            ], dtype=torch.float, device=device)  
        
    elif multi_view_option == 'rand_phi_each_step_along_azi_for_one_fixed_iter':
        # Check if it's time to update the phis value
        if iteration - phis_state["last_update_iteration"] >= phis_state["update_interval"] or phis_state["last_phis"] is None:
            # Generate new phis and update the state
            phis = torch.rand(size, device=device) * (np.deg2rad(phi_range[1]) - np.deg2rad(phi_range[0])) + np.deg2rad(phi_range[0])
            phis_state["last_phis"] = phis
            phis_state["last_update_iteration"] = iteration
        else:
            phis = phis_state["last_phis"] # Use the last stored phis value

    elif multi_view_option == 'rand_phi_each_step_along_azi_long_short_update_intervals':
        # Usage within the update logic
        if phis_state["last_phis"] is not None:
            # Determine if the last phis value is a side view
            if is_side_view(phis_state["last_phis"], device=device):
                current_update_interval = phis_state["long_update_interval"]
            else:
                current_update_interval = phis_state["short_update_interval"]
        else:
            # No last_phis means start with a long interval assuming first is a side view
            current_update_interval = phis_state["long_update_interval"]

        # Check if it's time to update the phis value
        if iteration > 0 and iteration % current_update_interval == 0:
            # Generate new phis and update the state
            phis = torch.rand(size, device=device) * (np.deg2rad(phi_range[1]) - np.deg2rad(phi_range[0])) + np.deg2rad(phi_range[0])
            phis_state["last_phis"] = phis
            phis_state["last_update_iteration"] = iteration
        else:
            # Use the last stored phis value
            phis = phis_state["last_phis"]
    
    elif multi_view_option == 'alternate_2_side_views_each_step_along_azimuth':
        if iteration > 0 and iteration % phis_state["update_interval"] == 0:
            # Generate new phis and update the state
            phis = get_2_alternating_phi(device)
            phis_state["last_phis"] = phis
            phis_state["last_update_iteration"] = iteration
        else:
            # Use the last stored phis value
            phis = phis_state["last_phis"]
        
    elif multi_view_option == 'alternate_4_side_views_each_step_along_azimuth':
        phis = get_4_alternating_phi(device)
        
    elif multi_view_option == "get_4_alternating_phi_45_degree_apart":
        phis = get_4_alternating_phi_45_degree_apart(device=device, iteration=iteration, update_interval=update_interval)
        
    elif multi_view_option == "get_2_alternating_phi_45_degree_apart":
        phis = get_2_alternating_phi_45_degree_apart(device=device, update_interval=update_interval)
        
    elif multi_view_option == 'multiple_random_phi_in_batch':                       
        phi_range = np.deg2rad(phi_range)
        # For azimuth rotation (phi), will create a sequence of values within the specified range
        # Do not include endpoint (as in np.linspace) to avoid duplicate values
        phis = torch.linspace(phi_range[0], phi_range[1], steps=size+1, device=device)[:size]
        
    # Keeping theta (elevation angle) constant
    thetas = torch.full((size,), theta, device=device)
    # targets = torch.zeros(size, 3, device=device)
    poses, dirs = poses_helper_func(size, device, phis, thetas, radius_range=[radius, radius])
    return poses, dirs


# ---------- BELOW FUNCTIONS ARE NOT USED IN THE CURRENT IMPLEMENTATION -------------

def rand_poses(size, device, theta_range=[0, 120], phi_range=[0, 360], uniform_sphere_rate=0.5, radius_range=[2.5, 2.5]):
    ''' generate random poses from an orbit camera
    Args:
        size: batch size of generated poses.
        device: where to allocate the output.
        theta_range: [min, max], should be in [0, pi]
        phi_range: [min, max], should be in [0, 2 * pi]
    Return:
        poses: [size, 4, 4]
    '''

    theta_range = np.deg2rad(theta_range)
    phi_range = np.deg2rad(phi_range)
    
    phis = torch.rand(size, device=device) * (phi_range[1] - phi_range[0]) + phi_range[0]
    
    if random.random() < uniform_sphere_rate:
        # based on http://corysimon.github.io/articles/uniformdistn-on-sphere/
        # acos takes in [-1, 1], first convert theta range to fit in [-1, 1] 
        theta_range = torch.from_numpy(np.array(theta_range)).to(device)
        theta_amplitude_range = torch.cos(theta_range)
        # sample uniformly in amplitude space range
        thetas_amplitude = torch.rand(size, device=device) * (theta_amplitude_range[1] - theta_amplitude_range[0]) + theta_amplitude_range[0]
        # convert back
        thetas = torch.acos(thetas_amplitude)
    else:
        thetas = torch.rand(size, device=device) * (theta_range[1] - theta_range[0]) + theta_range[0]
    
    poses, dirs = poses_helper_func(size, device, phis, thetas, radius_range=radius_range)
    
    return poses, dirs


# # Added for debugging purpose
# def poses_helper_func_single_view(size, device, phis, thetas, radius_range=[1, 1], angle_overhead=30, angle_front=60, phi_offset=0, jitter=False, cam_z_offset=12, return_dirs=True):

#     angle_overhead = np.deg2rad(angle_overhead)

#     angle_front = np.deg2rad(angle_front)
    
#     radius = torch.rand(size, device=device) * (radius_range[1] - radius_range[0]) + radius_range[0]

#     targets = torch.zeros(size, 3, device=device)

#     centers = -torch.stack([
#         radius * torch.sin(thetas) * torch.sin(phis),
#         radius * torch.cos(thetas),
#         radius * torch.sin(thetas) * torch.cos(phis),
#     ], dim=-1) # [B, 3]
    
#     # jitters
#     if jitter:
#         centers = centers + (torch.rand_like(centers) * 0.2 - 0.1)
#         targets = targets + torch.randn_like(centers) * 0.2
    
#     # lookat
#     forward_vector = safe_normalize(targets - centers)
#     up_vector = torch.FloatTensor([0, 1, 0]).to(device).unsqueeze(0).repeat(size, 1)
#     right_vector = safe_normalize(torch.cross(up_vector, forward_vector, dim=-1))
    
#     if jitter:
#         up_noise = torch.randn_like(up_vector) * 0.02
#     else:
#         up_noise = 0
    
#     up_vector = safe_normalize(torch.cross(forward_vector, right_vector, dim=-1) + up_noise)

#     poses = torch.stack([right_vector, up_vector, forward_vector], dim=-1)
#     radius = radius[..., None] - cam_z_offset

#     translations = torch.cat([torch.zeros_like(radius), torch.zeros_like(radius), radius], dim=-1) # Original

#     poses = torch.cat([poses.view(-1, 9), translations], dim=-1)

#     if return_dirs:
#         dirs = get_view_direction(thetas, phis, angle_overhead, angle_front, phi_offset=phi_offset)
#         dirs = view_direction_id_to_text(dirs)
#     else:
#         dirs = None

#     return poses, dirs

# def poses_along_azimuth_single_view(size, device, theta=90, phi_range=[0, 360]):
#     ''' generate random poses from an orbit camera along uniformly distributed azimuth and fixed elevation
#     Args:
#         size: batch size of generated poses.
#         device: where to allocate the output.
#         theta: is a constant                          
#         phi_range: [min, max], should be in [0, 2 * pi]
#     Return:
#         poses: [size, 4, 4]
#     '''
#     theta = np.deg2rad(theta)                   
#     phi_range = np.deg2rad(phi_range)
    
#     # For azimuth rotation (phi), we will create a sequence of values within the specified range
#     phis = torch.linspace(phi_range[0], phi_range[1], steps=size, device=device)
    
#     # Keeping theta (elevation angle) constant
#     thetas = torch.full((size,), theta, device=device)
#     # targets = torch.zeros(size, 3, device=device)
#     poses, dirs = poses_helper_func_single_view(size, device, phis, thetas)

#     return poses, dirs