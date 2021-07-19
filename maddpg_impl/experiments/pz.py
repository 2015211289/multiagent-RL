from pettingzoo.atari import tennis_v2
import supersuit
import numpy as np

# observations = env.reset()
# while True:
#     actions = {}
#     for agent in env.agents:
#         actions[agent] = env.action_spaces[agent].sample()
        
#     observations, rewards, dones, infos = env.step(actions)
#     env.render(mode='human')
#     if(all(dones.values())):
#         observations = env.reset()

# env.close()

def create_env():
    env = tennis_v2.parallel_env(obs_type='ram',full_action_space=False,auto_rom_install_path=None)
    # as per openai baseline's MaxAndSKip wrapper, maxes over the last 2 frames
    # to deal with frame flickering
    env = supersuit.max_observation_v0(env, 2)

    # repeat_action_probability is set to 0.25 to introduce non-determinism to the system
    env = supersuit.sticky_actions_v0(env, repeat_action_probability=0.25)

    # skip frames for faster processing and less control
    # to be compatable with gym, use frame_skip(env, (2,5))
    env = supersuit.frame_skip_v0(env, 4)

    # downscale observation for faster processing
    # env = supersuit.resize_v0(env, 84, 84)

    # allow agent to see everything on the screen despite Atari's flickering screen problem
    env = supersuit.frame_stack_v1(env, 4)
    return env



def step(actions,env):
    #TODO: 确认顺序
    targetActions = {}
    i=0
    for agent in env.possible_agents:
        if not isinstance(actions[i],int):
            targetActions[agent] = np.unravel_index(np.argmax(actions[i]),actions[i].shape)[0]
        else:
            targetActions[agent] = actions[i]
        i+=1
    
    observations, rewards, dones, infos = env.step(targetActions)
    # env.render()

    obs_n=[]
    rew_n=[]
    done_n=[]
    info_n=[]
    for agent in env.possible_agents:
        obs_n.append(observations[agent])
        rew_n.append(rewards[agent])
        done_n.append(dones[agent])
        info_n.append(infos[agent])

    return obs_n,rew_n,done_n,info_n
  