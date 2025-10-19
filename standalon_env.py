import time
import mujoco
import mujoco.glfw
from stable_baselines3 import PPO
from unitree_env import UnitreeEnv
import numpy as np
import argparse

# --- Helper function to create a rotation matrix for the arrow ---
def look_at(vec):
    vec = vec / np.linalg.norm(vec)
    up = np.array([0, 0, 1])
    if np.abs(np.dot(vec, up)) > 0.999:
        up = np.array([0, 1, 0])
    
    right = np.cross(up, vec)
    right /= np.linalg.norm(right)
    
    up = np.cross(vec, right)
    up /= np.linalg.norm(up)
    
    return np.array([right, up, vec]).T

# Geoms
GEOMS_TO_TOGGLE = [
    "RR_hip_geom", "RR_thigh_geom", "RR_calf_geom", "RR_foot_geom"
]

# --- CONFIGURATION ---
MODEL_PATH = "New_with_feet.zip"

# --- SETUP ---
print("Setting up the testing environment...")
env = UnitreeEnv(
    model_path='../unitree_mujoco/unitree_robots/go2/scene_ground.xml'
)
# CLI
parser = argparse.ArgumentParser(description="MuJoCo standalone env (optional RL model).")
parser.add_argument("--load-model", action="store_true", help="Load RL model and use it to act")
parser.add_argument("--model-path", type=str, default=MODEL_PATH, help="Path to RL model file")
args = parser.parse_args()

obs, info = env.reset()

# Load model conditionally
model = None
if args.load_model:
    try:
        model = PPO.load(args.model_path, env=env)
        print(f"Loaded RL model from: {args.model_path}")
    except Exception as e:
        print("Failed to load RL model:", e)
        model = None
else:
    print("Running without RL model (using zero actions).")

##
# --- Store original geom alphas for toggling ---
original_geom_alphas = {}
geom_ids_to_toggle = []
for geom_name in GEOMS_TO_TOGGLE:
    geom_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
    if geom_id != -1:
        original_alpha = env.model.geom_rgba[geom_id][3]
        original_geom_alphas[geom_id] = original_alpha
        geom_ids_to_toggle.append(geom_id)
        print(f"Found geom to toggle: {geom_name} (ID: {geom_id})")
    else:
        print(f"WARNING: Geom not found: {geom_name}")

# --- Manual Viewer Setup ---
mujoco.glfw.glfw.init()
window = mujoco.glfw.glfw.create_window(1200, 900, "Go2 Simulation", None, None)
mujoco.glfw.glfw.make_context_current(window)
mujoco.glfw.glfw.swap_interval(1)

cam = mujoco.MjvCamera()
opt = mujoco.MjvOption()
scn = mujoco.MjvScene(env.model, maxgeom=10000)
ctx = mujoco.MjrContext(env.model, mujoco.mjtFontScale.mjFONTSCALE_150)


# --- Camera Control Variables & Callbacks ---
button_left = False
button_middle = False
button_right = False
lastx = 0
lasty = 0

def mouse_button(window, button, action, mods):
    global button_left, button_middle, button_right
    if button == mujoco.glfw.glfw.MOUSE_BUTTON_LEFT:
        button_left = (action == mujoco.glfw.glfw.PRESS)
    elif button == mujoco.glfw.glfw.MOUSE_BUTTON_MIDDLE:
        button_middle = (action == mujoco.glfw.glfw.PRESS)
    elif button == mujoco.glfw.glfw.MOUSE_BUTTON_RIGHT:
        button_right = (action == mujoco.glfw.glfw.PRESS)

def mouse_move(window, xpos, ypos):
    global lastx, lasty, button_left, button_middle, button_right
    dx = xpos - lastx
    dy = ypos - lasty
    lastx, lasty = xpos, ypos
    if not (button_left or button_middle or button_right):
        return
    width, height = mujoco.glfw.glfw.get_window_size(window)
    action = 0
    if button_right:
        action = mujoco.mjtMouse.mjMOUSE_MOVE_H
    elif button_left:
        action = mujoco.mjtMouse.mjMOUSE_ROTATE_H
    mujoco.mjv_moveCamera(env.model, action, dx/width, dy/height, scn, cam)

def scroll(window, xoffset, yoffset):
    mujoco.mjv_moveCamera(env.model, mujoco.mjtMouse.mjMOUSE_ZOOM, 0, -0.05 * yoffset, scn, cam)

# Register the callbacks
mujoco.glfw.glfw.set_mouse_button_callback(window, mouse_button)
mujoco.glfw.glfw.set_cursor_pos_callback(window, mouse_move)
mujoco.glfw.glfw.set_scroll_callback(window, scroll)


print("\n" + "="*50)
print("Simulation is ready.")
print("CONTROLS:")
print("- PRESS SPACE to pause/play.")
print("- PRESS F to toggle contact forces on/off.")
print("- PRESS G to toggle contact sensor forces on/off.")
print("="*50 + "\n")

# --- Control State Variables ---
paused = True
show_forces = True
show_sensor_forces = True
hide_robot_geoms = False
opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = show_forces

env.model.vis.scale.forcewidth = 0.05
env.model.vis.map.force = 0.005
env.model.vis.rgba.contactforce = [1.0, 0.0, 0.0, 1.0] # Bright Red


# --- CUSTOM VISUALIZATION LOGIC ---
force_scale = 0.005
sensor_force_scale = 0.01
arrow_rgba = np.array([1.0, 0.0, 0.0, 1.0], dtype=np.float32)
sensor_arrow_rgba = np.array([1.0, 1.0, 0.0, 1.0], dtype=np.float32)
arrow_radius = 0.01

site_rgba = np.array([0.0, 0.5, 1.0, 1.0], dtype=np.float32) # Blue
site_size = np.array([0.02, 0.02, 0.02]) # Sphere size


# --- Input State for Debouncing ---
last_space_state = mujoco.glfw.glfw.RELEASE
last_f_state = mujoco.glfw.glfw.RELEASE
last_g_state = mujoco.glfw.glfw.RELEASE
last_h_state = mujoco.glfw.glfw.RELEASE

# --- Main Simulation Loop ---
try:
    while not mujoco.glfw.glfw.window_should_close(window):
        
        sim_start_time = time.time()

        # --- Simulation Step (only if not paused) ---
        if not paused:
            if model is not None:
                action, _states = model.predict(obs, deterministic=True)
            else:
                action = np.zeros(env.action_space.shape, dtype=np.float32)
            obs, reward, terminated, truncated, info = env.step(action)

            if terminated or truncated:
                print("\nEpisode finished. Resetting.")
                obs, info = env.reset()

        # --- GLFW Keyboard Input Handling ---
        space_state = mujoco.glfw.glfw.get_key(window, mujoco.glfw.glfw.KEY_SPACE)
        f_state = mujoco.glfw.glfw.get_key(window, mujoco.glfw.glfw.KEY_F)
        g_state = mujoco.glfw.glfw.get_key(window, mujoco.glfw.glfw.KEY_G)
        h_state = mujoco.glfw.glfw.get_key(window, mujoco.glfw.glfw.KEY_H)

        if space_state == mujoco.glfw.glfw.PRESS and last_space_state == mujoco.glfw.glfw.RELEASE:
            paused = not paused
            print(f"\rSimulation {'PAUSED' if paused else 'PLAYING'}", end="")

        if f_state == mujoco.glfw.glfw.PRESS and last_f_state == mujoco.glfw.glfw.RELEASE:
            show_forces = not show_forces
            opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = show_forces
            print(f"\rContact Forces {'ON' if show_forces else 'OFF'}    ", end="")

        if g_state == mujoco.glfw.glfw.PRESS and last_g_state == mujoco.glfw.glfw.RELEASE:
            show_sensor_forces = not show_sensor_forces
            print(f"\rSensor Contact Forces {'ON' if show_sensor_forces else 'OFF'}    ", end="")
        if h_state == mujoco.glfw.glfw.PRESS and last_h_state == mujoco.glfw.glfw.RELEASE:
            hide_robot_geoms = not hide_robot_geoms
            
            for geom_id in geom_ids_to_toggle:
                if hide_robot_geoms:
                    # Set alpha to 0 (invisible)
                    env.model.geom_rgba[geom_id][3] = 0.0
                else:
                    # Restore original alpha
                    env.model.geom_rgba[geom_id][3] = original_geom_alphas[geom_id]
            
            print(f"\rRobot Geoms {'HIDDEN' if hide_robot_geoms else 'VISIBLE'}    ", end="")

        last_space_state = space_state
        last_f_state = f_state
        last_g_state = g_state
        last_h_state = h_state

        # --- Rendering ---
        viewport_width, viewport_height = mujoco.glfw.glfw.get_framebuffer_size(window)
        viewport = mujoco.MjrRect(0, 0, viewport_width, viewport_height)

        # **** THIS IS THE KEY CHANGE ****
        # First, update the scene with the robot's geometry
        mujoco.mjv_updateScene(env.model, env.data, opt, None, cam, mujoco.mjtCatBit.mjCAT_ALL, scn)

        # Then, add all custom visualization geoms on top of the existing scene
        
        # --- DEBUG: Draw a big arrow on the robot's head ---
        if scn.ngeom < scn.maxgeom:
            geom = scn.geoms[scn.ngeom]
            base_pos = env.data.body('base_link').xpos
            arrow_pos = base_pos + np.array([0, 0, 0.2])
            arrow_size = np.array([0.05, 0.05, 0.3]) # radius1, radius2, length
            arrow_rgba_head = np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32) # Green
            up_direction = np.array([0, 0, 1])
            
            mujoco.mjv_initGeom(
                geom,
                type=mujoco.mjtGeom.mjGEOM_ARROW,
                size=arrow_size,
                pos=arrow_pos,
                mat=look_at(up_direction).flatten(),
                rgba=arrow_rgba_head
            )
            scn.ngeom += 1
        # --- END DEBUG ---


        # --- Sensor and Site Visualization ---
        if show_sensor_forces:
            for foot_name in ["FL", "FR", "RL", "RR"]:
                site_name = f'{foot_name}_site'
                
                # Draw Blue Sphere at Site Location
                if scn.ngeom < scn.maxgeom:
                    geom = scn.geoms[scn.ngeom]
                    site_pos = env.data.site(site_name).xpos
                    mujoco.mjv_initGeom(
                        geom,
                        type=mujoco.mjtGeom.mjGEOM_SPHERE,
                        size=site_size,
                        pos=site_pos,
                        mat=np.identity(3).flatten(),
                        rgba=site_rgba
                    )
                    scn.ngeom += 1

                # Draw Force Vector Arrow
                force_vector = env.data.sensor(f'{foot_name}_foot_force').data.copy()
                force_magnitude = np.linalg.norm(force_vector)
                
                if force_magnitude > 1.0:
                    start_pos = env.data.site(site_name).xpos
                    direction_vector = -force_vector / force_magnitude
                    
                    if scn.ngeom < scn.maxgeom:
                        geom = scn.geoms[scn.ngeom]
                        
                        mujoco.mjv_initGeom(
                            geom,
                            type=mujoco.mjtGeom.mjGEOM_ARROW,
                            size=np.array([arrow_radius, arrow_radius, force_magnitude * sensor_force_scale]),
                            pos=start_pos,
                            mat=look_at(direction_vector).flatten(),
                            rgba=sensor_arrow_rgba
                        )
                        scn.ngeom += 1

        # Finally, render the completed scene
        mujoco.mjr_render(viewport, scn, ctx)
        
        mujoco.glfw.glfw.swap_buffers(window)
        mujoco.glfw.glfw.poll_events()
        
        # --- Sync with simulation time ---
        time_until_next_step = env.model.opt.timestep - (time.time() - sim_start_time)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)


except KeyboardInterrupt:
    print("\nSimulation stopped by user.")
finally:
    mujoco.glfw.glfw.terminate()
    env.close()