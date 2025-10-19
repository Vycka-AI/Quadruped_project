import mujoco
import mujoco.viewer
# --- Configuration ---
# Use the new scene file that welds the robot's base
MODEL_XML_PATH = '../unitree_mujoco/unitree_robots/go2/scene_suspended.xml'

def main():
    """
    Launches the MuJoCo viewer for manual control of the Go2 robot.
    """
    # Load the model and data
    try:
        model = mujoco.MjModel.from_xml_path(MODEL_XML_PATH)
        data = mujoco.MjData(model)
    except FileNotFoundError:
        print(f"Error: XML file not found at '{MODEL_XML_PATH}'")
        print("Please make sure 'scene_suspended.xml' and 'go2.xml' are in the correct directory.")
        return


    # --- Print a detailed help message to the console ---
    print("\n" + "="*60)
    print("      Go2 Manual Control Simulator Launched")
    print("="*60)
    print("\nHow to control the robot in the simulation window:")
    
    print("\n--- Method 1: GUI Sliders (Precise Control) ---")
    print("1. Find the 'Control' window in the top-left of the GUI.")
    print("2. If it's collapsed, click the small triangle to expand it.")
    print("3. You will see 12 sliders, one for each joint motor (e.g., 'FR_hip', 'FL_thigh').")
    print("4. Drag these sliders to manually set the target angle for each joint.")

    print("\n--- Method 2: Mouse Dragging (Applying Forces) ---")
    print("1. Hold down the [CTRL] key.")
    print("2. Right-click and hold on any part of the robot (e.g., a foot or a thigh).")
    print("3. Drag the mouse to apply a force and pull the robot part around.")
    print("   (This is great for testing stability and inverse kinematics).")

    print("\n--- Other Viewer Controls ---")
    print("- PAUSE/PLAY:         Press [SPACE]")
    print("- RESET SIMULATION:   Press [BACKSPACE]")
    print("- SLOW MOTION:        Hold [CTRL] and press the down arrow key")
    print("- CAMERA (ROTATE):    Left-click and drag")
    print("- CAMERA (ZOOM):      Scroll wheel OR Right-click and drag up/down")
    print("- CAMERA (PAN):       Middle-click and drag")
    print("\nClose the window or press [ESC] to quit.")
    print("="*60 + "\n")

    # --- Launch the interactive viewer ---
    # The launch() function creates a window, handles all user input,
    # and runs the simulation loop.
    mujoco.viewer.launch(model, data)

if __name__ == "__main__":
    main()