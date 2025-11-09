from copied_isaac import UnitreeEnv
import numpy as np

if __name__ == "__main__":
    
    # --- !! IMPORTANT !! ---
    # --- SET YOUR MODEL PATH HERE ---
    XML_MODEL_PATH = "../unitree_mujoco/unitree_robots/go2/scene_ground.xml" # <--- UPDATE THIS
    # -----------------------
    
    print("Launching CPG-controlled quadruped in MuJoCo...")
    
    try:
        env = UnitreeEnv(model_path=XML_MODEL_PATH, 
                         render_mode="human", 
                         frame_skip=4) # frame_skip=4 -> dt=0.008s
        
        obs, info = env.reset(seed=42)
        
        # Create a dummy action that will be ignored
        dummy_action = np.zeros(env.action_space.shape)
        
        for i in range(10000): # Run for 10,000 steps
            obs, reward, terminated, truncated, info = env.step(dummy_action)
            
            if terminated or truncated:
                print(f"Episode finished after {i+1} steps.")
                obs, info = env.reset()
                
            # Optional: sleep to run in real-time
            # time.sleep(0.002) 

    except FileNotFoundError:
        print(f"Error: Model file not found at '{XML_MODEL_PATH}'")
        print("Please update the 'XML_MODEL_PATH' variable in the __main__ block.")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if 'env' in locals():
            env.close()
            print("Simulation closed.")