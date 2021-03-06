import numpy as np
import torch
import gym
import argparse
import os

import MATD3.utils as utils
import MATD3.TD3 as TD3
import MATD3.OurDDPG as OurDDPG
import MATD3.DDPG as DDPG

# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policys, env_name, seed, arg, eval_episodes=10):
    eval_env = gym.make(env_name)
    eval_env.seed(seed + 100)

    avg_reward = 0.
    for _ in range(eval_episodes):
        state, done = eval_env.reset(), False
        while not done:
            
            action = policy.select_action(np.array(state))
            state, reward, done, _ = eval_env.step(action)
            avg_reward += reward

    avg_reward /= eval_episodes

    print("---------------------------------------")
    print("Evaluation over {} episodes: {:.3f}".format(eval_episodes,avg_reward))
    print("---------------------------------------")
    return avg_reward

def make_env(scenario_name, arglist, benchmark=False):
    from multiagent.environment import MultiAgentEnv
    import multiagent.scenarios as scenarios

    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # create world
    world = scenario.make_world()
    try:
        arglist.num_adversaries = len(scenario.adversaries(world))
    except:
        arglist.num_adversaries = 0
        arglist.reward_shaping_adv = False
    print("adversary agents number is {}".format(arglist.num_adversaries))
    # create multiagent environment
    if benchmark:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data)
    else:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
    return env


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="TD3")                  # Policy name (TD3, DDPG or OurDDPG)
    parser.add_argument("--env", default="simple")          # OpenAI gym environment name
    parser.add_argument("--seed", default=0, type=int)              # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--start_timesteps", default=25e3, type=int)# Time steps initial random policy is used
    parser.add_argument("--eval_freq", default=5e3, type=int)       # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=1e6, type=int)   # Max time steps to run environment
    parser.add_argument("--expl_noise", default=0.1)                # Std of Gaussian exploration noise
    parser.add_argument("--batch_size", default=256, type=int)      # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99)                 # Discount factor
    parser.add_argument("--tau", default=0.005)                     # Target network update rate
    parser.add_argument("--policy_noise", default=0.2)              # Noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.5)                # Range to clip target policy noise
    parser.add_argument("--policy_freq", default=2, type=int)       # Frequency of delayed policy updates
    parser.add_argument("--save_model", action="store_true")        # Save model and optimizer parameters
    parser.add_argument("--load_model", default="")                 # Model load file name, "" doesn't load, "default" uses file_name
    parser.add_argument("--max_episodicsteps", default=25, type=int)  # Max episodic step of each episode
    args = parser.parse_args()

    file_name = "{}_{}_{}".format(args.policy,args.env,args.seed)
    print("---------------------------------------")
    print("Policy: {}, Env: {}, Seed: {}".format(args.policy,args.env,args.seed))
    print("---------------------------------------")

    if not os.path.exists("./results"):
        os.makedirs("./results")

    if args.save_model and not os.path.exists("./models"):
        os.makedirs("./models")

    # env = gym.make(args.env)
    env = make_env(args.env,args)

    # Set seeds
    env.seed(args.seed)
    # env.action_space.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # continous space
    # state_dim = env.observation_space.shape[0]
    # action_dim = env.action_space.shape[0] 
    # max_action = float(env.action_space.high[0])

    obs_shape_n = [env.observation_space[i].shape[0] for i in range(env.n)]
    action_shape_n = [env.action_space[i].n for i in range(env.n)]
    max_action_n = [1 for i in range(env.n)] # MPEs have no high attribute
    num_adversaries = args.num_adversaries

    kwargs = {
        "state_dim": obs_shape_n,
        "action_dim": action_shape_n,
        "max_action": max_action_n,
        "discount": args.discount,
        "tau": args.tau,
    }

    policys = []

    for i in range(env.n):
        # Initialize policy
        if args.policy == "TD3":
            # Target policy smoothing is scaled wrt the action scale
            kwargs["policy_noise"] = [args.policy_noise * x for x in max_action_n]
            kwargs["noise_clip"] = [args.noise_clip * x for x in max_action_n]
            kwargs["policy_freq"] = args.policy_freq
            policy = TD3.TD3(i,**kwargs)
            policys.append(policy)
        elif args.policy == "OurDDPG":
            policy = OurDDPG.DDPG(**kwargs)
        elif args.policy == "DDPG":
            policy = DDPG.DDPG(**kwargs)

    if args.load_model != "":
        policy_file = file_name if args.load_model == "default" else args.load_model
        policy.load("./models/{}".format(policy_file))
    
    # Evaluate untrained policy
    evaluations = [eval_policy(policys, args.env, args.seed)]

    state, done = env.reset(), False
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0

    for t in range(int(args.max_timesteps)):
        
        episode_timesteps += 1

        # Select action randomly or according to policy
        if t < args.start_timesteps:
            action = env.action_space.sample()
        else:
            action = (
                policy.select_action(np.array(state))
                + [np.random.normal(0, max_action_n[i] * args.expl_noise, size=action_shape_n[i]) 
                for i in range(env.n)]
            )

            for i in range(env.n):
                action[i].clip(-max_action_n[i],max_action_n[i])

        # Perform action
        next_state, reward, done, _ = env.step(action) 
        done_bool = float(done) if episode_timesteps < env._max_episode_steps else 0

        # Store data in replay buffer
        replay_buffer.add(state, action, next_state, reward, done_bool)

        state = next_state
        episode_reward += reward

        # Train agent after collecting sufficient data
        if t >= args.start_timesteps:
            policy.train(replay_buffer, args.batch_size)

        if done: 
            # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
            print("Total T: {} Episode Num: {} Episode T: {} Reward: {:.3f}".format(
                t+1,episode_num+1,episode_timesteps,episode_reward
            ))
            # Reset environment
            state, done = env.reset(), False
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1 

        # Evaluate episode
        if (t + 1) % args.eval_freq == 0:
            evaluations.append(eval_policy(policy, args.env, args.seed))
            np.save("./results/{}".format(file_name), evaluations)
            if args.save_model: policy.save("./models/{}".format(file_name))
