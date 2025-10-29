import time
import mujoco
import mujoco.glfw
import numpy as np
import os # Added to check if file exists

# --- Configuration ---
MODEL_XML_PATH = '../unitree_mujoco/unitree_robots/go2/scene_ground.xml' # *** CHANGED TO USE YOUR NEW XML ***
KEYFRAME_NAME = 'home' # Use the 'home' keyframe from your XML

# --- PD Controller Gains (Tune these!) ---
# These are EXAMPLE values. You'll need to tune them for good performance.
# Start low and gradually increase kp. Increase kd if it oscillates.
# For a free-floating robot, you might need higher gains than for a suspended one.
KP = 100.0 # Proportional gain (Stiffness) - Increased example value
KD = 5.0  # Derivative gain (Damping) - Increased example value

# --- Setup ---
if not os.path.exists(MODEL_XML_PATH):
    print(f"Error: Model file '{MODEL_XML_PATH}' not found.")
    print("Please make sure the XML file is in the same directory as the script.")
    exit(1)

try:
    # Load the model
    model = mujoco.MjModel.from_xml_path(MODEL_XML_PATH)
    data = mujoco.MjData(model)
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

# Get joint information
num_joints = model.nu # Number of actuators (should be 12)
if num_joints != 12:
     print(f"Warning: Expected 12 actuators, but found {num_joints}.")

joint_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt)]
actuator_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(model.nu)]

# Map actuator ctrl indices to joint qpos/qvel indices
# Assumes actuators directly map to the corresponding DOFs after the free joint
# Free joint has 7 qpos dimensions (pos + quat) and 6 qvel dimensions (linear + angular vel)
free_joint_qpos_dim = 7
free_joint_qvel_dim = 6

# Get the indices for the *actuated* joints in qpos and qvel arrays
actuator_joint_ids = [model.actuator_trnid[i, 0] for i in range(num_joints)] # Get the joint ID for each actuator
qpos_indices = []
qvel_indices = []
for jid in actuator_joint_ids:
    qpos_start = model.jnt_qposadr[jid]
    dof_start = model.jnt_dofadr[jid]
    # Assuming all actuated joints are 1-DOF hinge or slide joints
    qpos_indices.append(qpos_start)
    qvel_indices.append(dof_start)


print(f"Model has {num_joints} actuators.")
# print("Actuator names:", actuator_names)
# print("Corresponding qpos indices:", qpos_indices)
# print("Corresponding qvel indices:", qvel_indices)

# Check if the keyframe exists
key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, KEYFRAME_NAME)
if key_id == -1:
    print(f"Error: Keyframe '{KEYFRAME_NAME}' not found in the XML.")
    # Fallback: Use default pose (usually zeros for actuated joints)
    print("Using default initial pose for actuated joints (likely zeros).")
    target_qpos = np.zeros(num_joints)
    # Set initial pose only for actuated joints
    initial_qpos = data.qpos.copy() # Keep free joint default
    initial_qpos[qpos_indices] = target_qpos
    data.qpos[:] = initial_qpos
else:
    # Get the target joint positions from the keyframe
    key_qpos_full = model.key_qpos[key_id]
    target_qpos = key_qpos_full[qpos_indices] # Extract only the actuated joint values
    print(f"Target qpos from keyframe '{KEYFRAME_NAME}': {np.round(target_qpos, 3)}")
    # Set initial simulation state to the keyframe
    mujoco.mj_resetDataKeyframe(model, data, key_id)


# --- PD Gains Arrays ---
kp_array = np.full(num_joints, KP)
kd_array = np.full(num_joints, KD)

# Target velocity is zero for holding a pose
target_qvel = np.zeros(num_joints)

# --- Manual Viewer Setup ---
mujoco.glfw.glfw.init()
window = mujoco.glfw.glfw.create_window(1200, 900, "PD Control Test (go2.xml)", None, None)
mujoco.glfw.glfw.make_context_current(window)
mujoco.glfw.glfw.swap_interval(1)

cam = mujoco.MjvCamera()
opt = mujoco.MjvOption()
scn = mujoco.MjvScene(model, maxgeom=10000)
ctx = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)

# Default camera view
mujoco.mjv_defaultCamera(cam)
cam.azimuth = 120
cam.elevation = -20
cam.distance = 2.0 # Zoom in a bit closer
cam.lookat = data.qpos[:3] # Look at the base

# --- Camera Control Variables & Callbacks (Copied from standalon_env.py) ---
button_left = False
button_middle = False
button_right = False
lastx = 0
lasty = 0

def mouse_button(window, button, action, mods):
    global button_left, button_middle, button_right
    button_left = (mujoco.glfw.glfw.get_mouse_button(window, mujoco.glfw.glfw.MOUSE_BUTTON_LEFT) == mujoco.glfw.glfw.PRESS)
    button_middle = (mujoco.glfw.glfw.get_mouse_button(window, mujoco.glfw.glfw.MOUSE_BUTTON_MIDDLE) == mujoco.glfw.glfw.PRESS)
    button_right = (mujoco.glfw.glfw.get_mouse_button(window, mujoco.glfw.glfw.MOUSE_BUTTON_RIGHT) == mujoco.glfw.glfw.PRESS)

def mouse_move(window, xpos, ypos):
    global lastx, lasty, button_left, button_middle, button_right
    dx = xpos - lastx
    dy = ypos - lasty
    lastx, lasty = xpos, ypos
    if not (button_left or button_middle or button_right):
        return
    width, height = mujoco.glfw.glfw.get_window_size(window)
    action = mujoco.mjtMouse.mjMOUSE_NONE
    if button_right:
        action = mujoco.mjtMouse.mjMOUSE_MOVE_V # Use MOVE_V for vertical movement with right click
    elif button_left:
        action = mujoco.mjtMouse.mjMOUSE_ROTATE_H
    elif button_middle: # Add middle button panning
         action = mujoco.mjtMouse.mjMOUSE_MOVE_H
    mujoco.mjv_moveCamera(model, action, dx / width, dy / height, scn, cam)


def scroll(window, xoffset, yoffset):
    mujoco.mjv_moveCamera(model, mujoco.mjtMouse.mjMOUSE_ZOOM, 0, -0.05 * yoffset, scn, cam)

mujoco.glfw.glfw.set_mouse_button_callback(window, mouse_button)
mujoco.glfw.glfw.set_cursor_pos_callback(window, mouse_move)
mujoco.glfw.glfw.set_scroll_callback(window, scroll)

# --- Main Simulation Loop ---
try:
    print("\nSimulation running. Robot should try to reach and hold the 'home' pose.")
    print(f"Using KP={KP}, KD={KD}. Tune these values!")
    print("Close the window or press Ctrl+C to exit.")

    while not mujoco.glfw.glfw.window_should_close(window):
        sim_start_time = time.time()

        # --- PD Control Calculation ---
        # Get current state for actuated joints
        current_qpos = data.qpos[qpos_indices]
        current_qvel = data.qvel[qvel_indices]

        # Calculate position and velocity errors
        pos_error = target_qpos - current_qpos
        vel_error = target_qvel - current_qvel # target_qvel is zero

        # Calculate PD torques: torque = kp * pos_err + kd * vel_err
        pd_torques = kp_array * pos_error + kd_array * vel_error

        # Apply torques to actuators
        # Make sure torque limits from XML are respected (optional but good practice)
        ctrl_range = model.actuator_ctrlrange
        clamped_torques = np.clip(pd_torques, ctrl_range[:num_joints, 0], ctrl_range[:num_joints, 1])
        data.ctrl[:num_joints] = clamped_torques

        # --- Step Simulation ---
        mujoco.mj_step(model, data)

        # --- Rendering ---
        viewport_width, viewport_height = mujoco.glfw.glfw.get_framebuffer_size(window)
        viewport = mujoco.MjrRect(0, 0, viewport_width, viewport_height)

        # Update camera to follow the robot's base
        cam.lookat = data.qpos[:3]

        mujoco.mjv_updateScene(model, data, opt, None, cam, mujoco.mjtCatBit.mjCAT_ALL, scn)
        mujoco.mjr_render(viewport, scn, ctx)

        mujoco.glfw.glfw.swap_buffers(window)
        mujoco.glfw.glfw.poll_events()

        # --- Sync with simulation time (optional) ---
        time_until_next_step = model.opt.timestep - (time.time() - sim_start_time)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)

except KeyboardInterrupt:
    print("\nSimulation stopped by user.")
finally:
    if 'window' in locals() and window:
         mujoco.glfw.glfw.destroy_window(window)
    mujoco.glfw.glfw.terminate()