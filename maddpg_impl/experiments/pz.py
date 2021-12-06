from pettingzoo.atari import video_checkers_v3,combat_plane_v1,space_war_v1,joust_v2,surround_v1,ice_hockey_v1,pong_v2,entombed_cooperative_v2,double_dunk_v2,space_invaders_v1,maze_craze_v2,mario_bros_v2,wizard_of_wor_v2,basketball_pong_v2,boxing_v1,othello_v2,tennis_v2
import supersuit
import numpy as np

from experiments.plot import N

def create_env(env_name):
    env = None
    if env_name == "Basketball_Pong":
        env = basketball_pong_v2.parallel_env(obs_type='ram',full_action_space=False,auto_rom_install_path="/home/seth/anaconda3/envs/RL/lib/python3.7/site-packages/multi_agent_ale_py/roms")
    elif env_name == "Double_Dunk":
        env = double_dunk_v2.parallel_env(obs_type='ram',full_action_space=False,auto_rom_install_path="/home/seth/anaconda3/envs/RL/lib/python3.7/site-packages/multi_agent_ale_py/roms")
    elif env_name == "Space_Invaders":
        env = space_invaders_v1.parallel_env(alternating_control=False, moving_shields=True,
zigzaging_bombs=True, fast_bomb=True, invisible_invaders=False,obs_type='ram',full_action_space=True,auto_rom_install_path="/home/seth/anaconda3/envs/RL/lib/python3.7/site-packages/multi_agent_ale_py/roms")
    elif env_name == "Maze_Craze":
        env = maze_craze_v2.parallel_env(game_version="race",visibilty_level=0,obs_type='ram',full_action_space=True,auto_rom_install_path="/home/seth/anaconda3/envs/RL/lib/python3.7/site-packages/multi_agent_ale_py/roms")
    elif env_name == "Mario_Bros":
        env = mario_bros_v2.parallel_env(obs_type='ram',full_action_space=False,auto_rom_install_path="/home/seth/anaconda3/envs/RL/lib/python3.7/site-packages/multi_agent_ale_py/roms")
    elif env_name == "Wizard_of_Wor":
        env = wizard_of_wor_v2.parallel_env(obs_type='ram',full_action_space=False,auto_rom_install_path="/home/seth/anaconda3/envs/RL/lib/python3.7/site-packages/multi_agent_ale_py/roms")
    elif env_name =="Box":
        env = boxing_v1.parallel_env(obs_type='ram',full_action_space=False,auto_rom_install_path="/home/seth/anaconda3/envs/RL/lib/python3.7/site-packages/multi_agent_ale_py/roms")
    elif env_name == "Othello":
        env  = othello_v2.parallel_env(obs_type='ram',full_action_space=False,auto_rom_install_path="/home/seth/anaconda3/envs/RL/lib/python3.7/site-packages/multi_agent_ale_py/roms")
    elif env_name == "Tennis":
        env = tennis_v2.parallel_env(obs_type='ram',full_action_space=False,auto_rom_install_path="/home/seth/anaconda3/envs/RL/lib/python3.7/site-packages/multi_agent_ale_py/roms")
    elif env_name == "Pong":
        env = pong_v2.parallel_env(obs_type='ram',full_action_space=False,auto_rom_install_path="/home/seth/anaconda3/envs/RL/lib/python3.7/site-packages/multi_agent_ale_py/roms")
    elif env_name == "Entombed:Cooperative":
        env = entombed_cooperative_v2.parallel_env(obs_type='ram',full_action_space=False,auto_rom_install_path="/home/seth/anaconda3/envs/RL/lib/python3.7/site-packages/multi_agent_ale_py/roms")
    elif env_name == "Surround":
        env = surround_v1.parallel_env(obs_type='ram',full_action_space=False,auto_rom_install_path="/home/seth/anaconda3/envs/RL/lib/python3.7/site-packages/multi_agent_ale_py/roms")
    elif env_name == "Ice_Hockey":
        env = ice_hockey_v1.parallel_env(obs_type='ram',full_action_space=False,auto_rom_install_path="/home/seth/anaconda3/envs/RL/lib/python3.7/site-packages/multi_agent_ale_py/roms")
    elif env_name == "Joust":
        env = joust_v2.parallel_env(obs_type='ram',full_action_space=False,auto_rom_install_path="/home/seth/anaconda3/envs/RL/lib/python3.7/site-packages/multi_agent_ale_py/roms")
    elif env_name == "Space_War":
        env = space_war_v1.parallel_env(obs_type='ram',full_action_space=False,auto_rom_install_path="/home/seth/anaconda3/envs/RL/lib/python3.7/site-packages/multi_agent_ale_py/roms")
    elif env_name == "Combat_Plane":
        env = combat_plane_v1.parallel_env(game_version="jet", guided_missile=True, obs_type='ram',full_action_space=False,auto_rom_install_path="/home/seth/anaconda3/envs/RL/lib/python3.7/site-packages/multi_agent_ale_py/roms")
    elif env_name == "Video_Checkers":
        env = video_checkers_v3.parallel_env(obs_type='ram',full_action_space=False,auto_rom_install_path="/home/seth/anaconda3/envs/RL/lib/python3.7/site-packages/multi_agent_ale_py/roms")

    # as per openai baseline's MaxAndSKip wrapper, maxes over the last 2 frames
    # to deal with frame flickering
    env = supersuit.max_observation_v0(env, 2)

    # repeat_action_probability is set to 0.25 to introduce non-determinism to the system
    env = supersuit.sticky_actions_v0(env, repeat_action_probability=0.25)

    # skip frames for faster processing and less control
    # to be compatable with gym, use frame_skip(env, (2,5))
    env = supersuit.frame_skip_v0(env, 4)

    # downscale observation for faster processing
    # env = supersuit.resize_v0(env, 84,84)

    # allow agent to see everything on the screen despite Atari's flickering screen problem
    # env = supersuit.frame_stack_v1(env, 4)

    # avoid some agent died and action step exception
    env = supersuit.black_death_v2(env)
    return env



def step(actions,env):
    targetActions = {}
    for action,agent in actions:
        if not isinstance(actions,int):
            max_index = 0
            max = 0
            for i in range(action.shape[0]):
                if action[i]>max:
                    max = action[i]
                    max_index=i
            targetActions[agent] = max_index
        else:
            targetActions[agent] = action
    
    
    observations, rewards, dones, infos = env.step(targetActions)
  
    # env.render()

    obs_n=[]
    rew_n=[]
    done_n=[]
    info_n=[]

    bits=[]
    for _,agent in actions:
        # tmp = observations[agent]
        # for num in tmp:
        #     for n in uint8_to_bits(num):
        #         bits.append(n)
        obs_n.append(observations[agent])
        rew_n.append(rewards[agent])
        done_n.append(dones[agent])
        info_n.append(infos[agent])

    return obs_n,rew_n,done_n,info_n

def uint8_to_bits(number):
    result=[0] *8
    for i in range(8):
        result[7-i]=number%2
        number=number//2
    return result

def reset(env,agents):
    bits=[]
    observations = env.reset()
    obs_n=[]
    for agent in agents:
        # tmp = observations[agent]
        # for num in tmp:
        #     for n in uint8_to_bits(num):
        #         bits.append(n)
        # obs_n.append(np.array(bits))
        obs_n.append(observations[agent])
    
    return obs_n