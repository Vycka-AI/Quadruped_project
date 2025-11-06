import mujoco
import mujoco.viewer
import numpy as np
import time

class SuspendedVisualizer:
    """
    Loads a Go2 robot model, sets it to a specific suspended pose,
    and launches the MuJoCo viewer to display it.
    
    The simulation does not step forward; it just visualizes the static pose.
    """
    def __init__(self, model_path):
        # Load the MuJoCo model
        try:
            self.model = mujoco.MjModel.from_xml_path(model_path)
        except Exception as e:
            print(f"Error loading XML from {model_path}")
            print(f"Please make sure the path is correct.")
            print(f"Error details: {e}")
            raise
            
        self.data = mujoco.MjData(self.model)

        # --- Define the desired spawn pose ---
        self.spawn_base_pos = np.array([0.0, 0.0, 0.42])
        self.spawn_base_orn = np.array([1.0, 0.0, 0.0, 0.0]) # Default (w,x,y,z) quaternion

        # The dictionary of angles you provided
        spawn_joint_angles_map = {
            'FL_hip_joint': 0.1,   'RL_hip_joint': 0.1,
            'FR_hip_joint': -0.1,  'RR_hip_joint': -0.1,
            'FL_thigh_joint': 0.8, 'RL_thigh_joint': 1.0,
            'FR_thigh_joint': 0.8, 'RR_thigh_joint': 1.0,
            'FL_calf_joint': -1.5, 'RL_calf_joint': -1.5,
            'FR_calf_joint': -1.5, 'RR_calf_joint': -1.5,
        }

        # Create the ordered array for qpos[7:]
        # Assumes joint order: FL, FR, RL, RR
        self.default_joint_angles_ordered = np.array([
            spawn_joint_angles_map['FL_hip_joint'],   # 0.1
            spawn_joint_angles_map['FL_thigh_joint'], # 0.8
            spawn_joint_angles_map['FL_calf_joint'],  # -1.5
            spawn_joint_angles_map['FR_hip_joint'],   # -0.1
            spawn_joint_angles_map['FR_thigh_joint'], # 0.8
            spawn_joint_angles_map['FR_calf_joint'],  # -1.5
            spawn_joint_angles_map['RL_hip_joint'],   # 0.1
            spawn_joint_angles_map['RL_thigh_joint'], # 1.0
            spawn_joint_angles_map['RL_calf_joint'],  # -1.5
            spawn_joint_angles_map['RR_hip_joint'],   # -0.1
            spawn_joint_angles_map['RR_thigh_joint'], # 1.0
            spawn_joint_angles_map['RR_calf_joint'],  # -1.5
        ])
        
        # Set the robot to this pose
        self.reset_pose()

    def reset_pose(self):
        """Sets the robot's qpos and qvel to the defined spawn pose."""
        # 1. Reset the entire simulation
        mujoco.mj_resetData(self.model, self.data)

        # 2. Set the base position (indices 0-2)
        self.data.qpos[0:3] = self.spawn_base_pos

        # 3. Set the base orientation (indices 3-6)
        self.data.qpos[3:7] = self.spawn_base_orn

        # 4. Set the 12 joint angles (indices 7-18)
        self.data.qpos[7:] = self.default_joint_angles_ordered

        # 5. Set all velocities to zero
        self.data.qvel[:] = 0.0
        
        # 6. Forward the simulation to apply these changes
        mujoco.mj_forward(self.model, self.data)

    def run(self):
        """Launches the viewer and keeps it open."""
        print("Launching MuJoCo viewer...")
        print("The robot is suspended in its initial pose.")
        print("Close the viewer window to exit the script.")
        
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            while viewer.is_running():
                # We don't call mj_step, so the simulation doesn't advance
                viewer.sync()
                # Sleep a little to not hog the CPU
                time.sleep(0.01)

# --- This is the main runnable part ---
if __name__ == "__main__":
    
    # !!! IMPORTANT !!!
    # !!! Replace this with the path to your Go2 XML file !!!
    MODEL_XML_PATH = '../unitree_mujoco/unitree_robots/go2/scene_suspended.xml'

    if MODEL_XML_PATH == 'YOUR_MODEL_PATH_HERE.xml':
        print("Error: Please update 'MODEL_XML_PATH' in the script to point to your Go2 XML file.")
    else:
        try:
            visualizer = SuspendedVisualizer(MODEL_XML_PATH)
            visualizer.run()
        except Exception as e:
            print(f"\nAn error occurred: {e}")
