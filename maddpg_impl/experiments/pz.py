from pettingzoo.atari import pong_v2,entombed_cooperative_v2,double_dunk_v2,space_invaders_v1,maze_craze_v2,mario_bros_v2,wizard_of_wor_v2,basketball_pong_v2,boxing_v1,othello_v2,tennis_v2
# import supersuit
import numpy as np

def create_env(env_name):
    env = None
    if env_name == "Basketball_Pong":
        env = basketball_pong_v2.parallel_env(obs_type='ram',full_action_space=False,auto_rom_install_path="/home/seth/anaconda3/envs/RL/lib/python3.7/site-packages/multi_agent_ale_py/roms")
    elif env_name == "Double_Dunk":
        env = double_dunk_v2.parallel_env(obs_type='ram',full_action_space=False,auto_rom_install_path="/home/seth/anaconda3/envs/RL/lib/python3.7/site-packages/multi_agent_ale_py/roms")
    elif env_name == "Space_Invaders":
        env = space_invaders_v1.parallel_env(obs_type='ram',full_action_space=False,auto_rom_install_path="/home/seth/anaconda3/envs/RL/lib/python3.7/site-packages/multi_agent_ale_py/roms")
    elif env_name == "Maze_Craze":
        env=maze_craze_v2.parallel_env(game_version="capture",visibilty_level=0,obs_type='ram',full_action_space=False,auto_rom_install_path="/home/seth/anaconda3/envs/RL/lib/python3.7/site-packages/multi_agent_ale_py/roms")
    elif env_name == "Mario_Bros":
        env=mario_bros_v2.parallel_env(obs_type='ram',full_action_space=False,auto_rom_install_path="/home/seth/anaconda3/envs/RL/lib/python3.7/site-packages/multi_agent_ale_py/roms")
    elif env_name == "Wizard_of_Wor":
        env = wizard_of_wor_v2.parallel_env(obs_type='ram',full_action_space=False,auto_rom_install_path="/home/seth/anaconda3/envs/RL/lib/python3.7/site-packages/multi_agent_ale_py/roms")
    elif env_name =="Box":
        env = boxing_v1.parallel_env(obs_type='ram',full_action_space=False,auto_rom_install_path="/home/seth/anaconda3/envs/RL/lib/python3.7/site-packages/multi_agent_ale_py/roms")
    elif env_name == "Othello":
        env  = othello_v2.parallel_env(obs_type='ram',full_action_space=False,auto_rom_install_path="/home/seth/anaconda3/envs/RL/lib/python3.7/site-packages/multi_agent_ale_py/roms")
    elif env_name == "Tennis":
        env = tennis_v2.parallel_env(obs_type='ram',full_action_space=False,auto_rom_install_path="/home/seth/anaconda3/envs/RL/lib/python3.7/site-packages/multi_agent_ale_py/roms")
    elif env_name == "Pong":
        env_name = pong_v2.parallel_env(obs_type='ram',full_action_space=False,auto_rom_install_path="/home/seth/anaconda3/envs/RL/lib/python3.7/site-packages/multi_agent_ale_py/roms")


    # as per openai baseline's MaxAndSKip wrapper, maxes over the last 2 frames
    # to deal with frame flickering
    # env = supersuit.max_observation_v0(env, 2)

    # repeat_action_probability is set to 0.25 to introduce non-determinism to the system
    # env = supersuit.sticky_actions_v0(env, repeat_action_probability=0.25)

    # skip frames for faster processing and less control
    # to be compatable with gym, use frame_skip(env, (2,5))
    # env = supersuit.frame_skip_v0(env, 4)

    # downscale observation for faster processing
    # env = supersuit.resize_v0(env, 84, 84)

    # allow agent to see everything on the screen despite Atari's flickering screen problem
    # env = supersuit.frame_stack_v1(env, 4)
    return env



def step(actions,env):
    targetActions = {}
    for action,agent in actions:
        if not isinstance(actions,int):
            targetActions[agent] = np.unravel_index(np.argmax(action),action.shape)[0]
        else:
            targetActions[agent] = action
    
    observations, rewards, dones, infos = env.step(targetActions)
    # env.render()

    obs_n=[]
    rew_n=[]
    done_n=[]
    info_n=[]
    for _,agent in actions:
        obs_n.append(observations[agent])
        rew_n.append(rewards[agent])
        done_n.append(dones[agent])
        info_n.append(infos[agent])

    return obs_n,rew_n,done_n,info_n
  