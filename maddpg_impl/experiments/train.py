import argparse
import numpy as np
import os
# use GPU or not
# if network is small and shallow, CPU may be faster than GPU
os.environ["CUDA_VISIBLE_DEVICES"]="-1" 
import tensorflow as tf
import time
import pickle

import maddpg.common.tf_util as U
from maddpg.trainer.maddpg import MADDPGAgentTrainer
import tensorflow.contrib.layers as layers
from reward_shaping.embedding_model import EmbeddingModel
from reward_shaping.config import Config
from multiagent.multi_discrete import MultiDiscrete

def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # Environment
    parser.add_argument("--scenario", type=str, default="simple_reference", help="name of the scenario script")
    parser.add_argument("--max-episode-len", type=int, default=500, help="maximum episode length")
    parser.add_argument("--num-episodes", type=int, default=500, help="number of episodes")
    parser.add_argument("--num-adversaries", type=int, default=1, help="number of adversaries")
    parser.add_argument("--good-policy", type=str, default="TD3", help="policy for good agents")
    parser.add_argument("--adv-policy", type=str, default="TD3", help="policy of adversaries")
    # Core training parameters
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate for Adam optimizer")
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    parser.add_argument("--batch-size", type=int, default=128, help="number of episodes to optimize at the same time")
    parser.add_argument("--num-units", type=int, default=128, help="number of units in the mlp")
    # Checkpointing
    parser.add_argument("--exp-name", type=str, default="test", help="name of the experiment")
    parser.add_argument("--save-dir", type=str, default="./policy/", help="directory in which training state and model should be saved")
    parser.add_argument("--save-rate", type=int, default=1, help="save model once every time this many episodes are completed")
    parser.add_argument("--load-dir", type=str, default="./policy/", help="directory in which training state and model are loaded")
    # Evaluation
    parser.add_argument("--restore", action="store_true", default=False)
    parser.add_argument("--display", action="store_true", default=False)
    parser.add_argument("--benchmark", action="store_true", default=False)
    parser.add_argument("--benchmark-iters", type=int, default=100000, help="number of iterations run for benchmarking")
    parser.add_argument("--benchmark-dir", type=str, default="./benchmark_files/", help="directory where benchmark data is saved")
    parser.add_argument("--plots-dir", type=str, default="./complex_game/", help="directory where plot data is saved")
    parser.add_argument("--reward-shaping-ag", action="store_true", default=False, help="whether enable reward shaping of agents")
    parser.add_argument("--reward-shaping-adv", action="store_true", default=False, help="whether enable reward shaping of adversaries")
    parser.add_argument("--policy_noise", default=0.2,type=float)      
    parser.add_argument("--noise_clip", default=0.2,type=float)
    parser.add_argument("--policy_freq", default=2, type=int)
    parser.add_argument("--pettingzoo", action="store_true", default=False)
    parser.add_argument("--start_timesteps", default=10240, type=int)


    return parser.parse_args()

def mlp_model(input, num_outputs, scope, reuse=False, num_units=64, rnn_cell=None):
    # This model takes as input an observation and returns values of all actions
    with tf.variable_scope(scope, reuse=reuse):
        out = input
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_units//2, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_outputs, activation_fn=None)
        return out

def make_env(scenario_name, arglist, benchmark=False):
    from multiagent.environment import MultiAgentEnv
    import multiagent.scenarios as scenarios
    from experiments.pz import create_env

    print("env is ",arglist.scenario)

    if arglist.pettingzoo:
        env = create_env(arglist.scenario)
        # arglist.num_adversaries = 0
        print("adversary agents number is {}".format(arglist.num_adversaries))
        return env

    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # create world
    world = scenario.make_world()
    try:
        arglist.num_adversaries = len(scenario.adversaries(world))
    except:
        if arglist.scenario == 'simple_push':
            arglist.num_adversaries = 1
        else:
            arglist.num_adversaries = 0
            arglist.reward_shaping_adv = False
    print("adversary agents number is {}".format(arglist.num_adversaries))
    # create multiagent environment
    if benchmark:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data)
    else:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
    return env

def get_trainers(env, num_adversaries, obs_shape_n, arglist, agents):
    trainers = []
    model = mlp_model
    trainer = MADDPGAgentTrainer
    if not arglist.pettingzoo:
        for i in range(num_adversaries):
            trainers.append(trainer(
                "agent_%d" % i, model, obs_shape_n, env.action_space, i, arglist,
                local_q_func=(arglist.adv_policy=='ddpg'),agent=None))
        for i in range(num_adversaries, env.n):
            trainers.append(trainer(
                "agent_%d" % i, model, obs_shape_n, env.action_space, i, arglist,
                local_q_func=(arglist.good_policy=='ddpg'),agent=None))
    else:
        trainers.append(trainer(
            "agent_%d" % 0, model, obs_shape_n, env.action_spaces.values(), 0, arglist,
            local_q_func=(arglist.adv_policy=='ddpg'),agent=agents[0]))
        
        trainers.append(trainer(
            "agent_%d" % 1, model, obs_shape_n, env.action_spaces.values(), 1, arglist,
            local_q_func=(arglist.good_policy=='ddpg'),agent=agents[1]))

    return trainers

def create_dirs(arglist):
    import os
    os.makedirs(os.path.dirname(arglist.benchmark_dir), exist_ok=True)
    os.makedirs(os.path.dirname(arglist.plots_dir), exist_ok=True)

def transform_obs_n(obs_n):
    import torch
    input = obs_n[0]
    for i in range(1, len(obs_n)):
        input = np.append(input, obs_n[i])
    return torch.from_numpy(input).float()

def train(arglist):
    with U.single_threaded_session():
        # Create environment
        env = make_env(arglist.scenario, arglist, arglist.benchmark)
        # Create agent trainers
        if not arglist.pettingzoo:
            obs_shape_n = [env.observation_space[i].shape for i in range(env.n)]
            action_shape_n = []
            for i in range(env.n):
                if hasattr(env.action_space[i],"n"):
                    action_shape_n.append(env.action_space[i].n)
                elif not isinstance(env.action_space[i],MultiDiscrete):
                    action_shape_n.append(env.action_space[i].shape)
                else:
                    num = 0
                    for j in range(len(env.action_space[i].high)):
                        num+=(env.action_space[i].high[j]-env.action_space[i].low[j]+1)
                    action_shape_n.append(num)
        else:
            agents = [agent for agent in (env.possible_agents)]
            obs_shape_n = [env.observation_spaces[agent].shape for agent in agents]
            action_shape_n = []
            for agent in agents:
                if hasattr(env.action_spaces[agent],"n"):
                    action_shape_n.append(env.action_spaces[agent].n)
                else:
                    action_shape_n.append(env.action_spaces[agent].shape)

        num_adversaries = min(99, arglist.num_adversaries)
        trainers = get_trainers(env, num_adversaries, obs_shape_n, arglist, agents)
        print('Using good policy {} and adv policy {}'.format(arglist.good_policy, arglist.adv_policy))

        # Initialize
        U.initialize()

        # Load previous results, if necessary
        if arglist.load_dir == "":
            arglist.load_dir = arglist.save_dir
        if arglist.restore or arglist.benchmark:
            print('Loading previous state...')
            U.load_state(arglist.load_dir)

        # create dirs for saving benchmark data and reward data
        create_dirs(arglist)

        episode_rewards = [0.0]  # sum of rewards for all agents
        episode_original_rewards = [0.0]  # sum of original rewards for all agents
        if not arglist.pettingzoo:
            agent_rewards = [[0.0] for _ in range(env.n)]  # individual agent reward
            agent_original_rewards = [[0.0] for _ in range(env.n)]  # individual original agent reward

        else:
            agent_rewards = [[0.0] for _ in env.possible_agents]
            agent_original_rewards = [[0.0] for _ in env.possible_agents]  # individual original agent reward
        final_ep_rewards = []  # sum of rewards for training curve
        final_ep_ag_rewards = []  # agent rewards for training curve
        agent_info = [[[]]]  # placeholder for benchmarking info
        saver = tf.train.Saver()
        
        if not arglist.pettingzoo:
            obs_n = env.reset()
        else:
            t = env.reset()
            obs_n=[]
            for agent in agents:
                obs_n.append(t[agent])

        episode_step = 0
        train_step = 0
        # two teams embedding network
        embedding_model_adv = EmbeddingModel(obs_size=obs_shape_n[0:num_adversaries], num_outputs=action_shape_n[0:num_adversaries])
        embedding_model_ag = EmbeddingModel(obs_size=obs_shape_n[num_adversaries:], num_outputs=action_shape_n[num_adversaries:])
        episodic_memory_adv = []
        episodic_memory_ag = []
        if arglist.reward_shaping_adv:
            episodic_memory_adv.append(embedding_model_adv.embedding(transform_obs_n(obs_n[0:num_adversaries])))
        if arglist.reward_shaping_ag:
            episodic_memory_ag.append(embedding_model_ag.embedding(transform_obs_n(obs_n[num_adversaries:])))

        t_start = time.time()
        print('Starting iterations...')
        while True:
            # get action: possibility distribution
            # env.render()
            action_n=[]
            if  train_step < arglist.start_timesteps and arglist.pettingzoo and not arglist.restore:
                for agent,shape in zip(trainers,action_shape_n):
                    action = np.random.rand(shape)
                    action = action / np.sum(action)
                    action_n.append((action,agent.agent))

            else:
                action_n = [(agent.action(obs),agent.agent) for agent, obs in zip(trainers,obs_n)]
            # environment step
            if not arglist.pettingzoo:
                new_obs_n, rew_n, done_n, info_n = env.step(action_n)
            else:
                from experiments.pz import step
                # 预防环境异常
                try:
                    new_obs_n, rew_n, done_n, info_n = step(action_n,env)
                except Exception:
                    t = env.reset()
                    obs_n=[]
                    for agent in agents:  
                        obs_n.append(t[agent])
                    continue

            original_rew_n = rew_n.copy()
            action_n = [action for action,agent in action_n]
            
            # add reward shaping
            if arglist.reward_shaping_adv == True:
                next_state_emb_adv = embedding_model_adv.embedding(transform_obs_n(new_obs_n[0:num_adversaries]))
                intrinsic_reward_adv = embedding_model_adv.compute_intrinsic_reward(episodic_memory_adv, next_state_emb_adv)
                episodic_memory_adv.append(next_state_emb_adv)
                for i in range(0,num_adversaries):
                    # can add life long curiosity
                    rew_n[i] += Config.beta * (1-len(episode_rewards)/arglist.num_episodes) *intrinsic_reward_adv

            if arglist.reward_shaping_ag == True:
                next_state_emb_ag = embedding_model_ag.embedding(transform_obs_n(new_obs_n[num_adversaries:]))
                intrinsic_reward_ag = embedding_model_ag.compute_intrinsic_reward(episodic_memory_ag, next_state_emb_ag)
                episodic_memory_ag.append(next_state_emb_ag)
                if not arglist.pettingzoo:
                    for i in range(num_adversaries,env.n):
                        rew_n[i] += Config.beta * (1-len(episode_rewards)/arglist.num_episodes) * intrinsic_reward_ag
                else:
                    for i in range(num_adversaries,len(env.possible_agents)):
                        rew_n[i] += Config.beta *(1-len(episode_rewards)/arglist.num_episodes) * intrinsic_reward_ag

            episode_step += 1
            done = all(done_n)
            terminal = (episode_step >= arglist.max_episode_len)
            # collect experience
            for i, agent in enumerate(trainers):
                agent.experience(obs_n[i], action_n[i], rew_n[i], new_obs_n[i], done_n[i], terminal)
            obs_n = new_obs_n

            for i, rew in enumerate(rew_n):
                episode_rewards[-1] += rew
                episode_original_rewards[-1] += original_rew_n[i]
                agent_rewards[i][-1] += rew
                agent_original_rewards[i][-1] += original_rew_n[i]

            if done or terminal:
                terminal = True
                # obs_n = env.reset()
                if not arglist.pettingzoo:
                    obs_n = env.reset()
                else:
                    t = env.reset()
                    obs_n=[]
                    for agent in agents:  
                        obs_n.append(t[agent])

                episode_step = 0
                episode_rewards.append(0)
                episode_original_rewards.append(0)
                for a in agent_rewards:
                    a.append(0)
                for a in agent_original_rewards:
                    a.append(0)
                agent_info.append([[]])

                # reset episode embedding network
                episodic_memory_adv.clear()
                # embedding_model_adv.lastReward=0
                episodic_memory_ag.clear()
                # embedding_model_ag.lastReward=0
                if arglist.reward_shaping_adv:
                    episodic_memory_adv.append(embedding_model_adv.embedding(transform_obs_n(obs_n[0:num_adversaries])))
                if arglist.reward_shaping_ag:
                    episodic_memory_ag.append(embedding_model_ag.embedding(transform_obs_n(obs_n[num_adversaries:])))

            # increment global step counter
            train_step += 1

            # for benchmarking learned policies
            if arglist.benchmark:
                for i, info in enumerate(info_n):
                    agent_info[-1][i].append(info_n['n'])
                if train_step > arglist.benchmark_iters and (done or terminal):
                    file_name = arglist.benchmark_dir + arglist.exp_name + '.pkl'
                    print('Finished benchmarking, now saving...')
                    with open(file_name, 'wb') as fp:
                        pickle.dump(agent_info[:-1], fp)
                    break
                continue

            # for displaying learned policies
            if arglist.display:
                # time.sleep(0.1)
                env.render()
                continue

            # update all trainers, if not in display or benchmark mode
            loss = None
            for agent in trainers:
                agent.preupdate()
            for agent in trainers:
                loss = agent.update(trainers, train_step)
            
            # train embedding network
            obs_n_train = []
            obs_next_n_train = []
            act_n_train = []
            embedding_loss_ag = None
            embedding_loss_adv = None
            if train_step > 0 and (arglist.reward_shaping_adv or arglist.reward_shaping_ag):
            
                if arglist.reward_shaping_adv == True and train_step > Config.train_episode_num:
                    for i in range(0,num_adversaries):
                        obs, act, rew, obs_next, done = trainers[i].sample(Config.train_episode_num)
                        obs_n_train.append(obs)
                        obs_next_n_train.append(obs_next)
                        act_n_train.append(act)

                    embedding_loss_adv = embedding_model_adv.train_model(obs_n_train,obs_next_n_train,act_n_train)

                if arglist.reward_shaping_ag == True and train_step > Config.train_episode_num:
                    obs_n_train = []
                    obs_next_n_train = []
                    act_n_train = []
                    n = 0
                    if not arglist.pettingzoo:
                        n= env.n
                    else:
                        n= len(env.possible_agents)
                    for i in range(num_adversaries,n):
                        obs, act, rew, obs_next, done = trainers[i].sample(Config.train_episode_num)
                        obs_n_train.append(obs)
                        obs_next_n_train.append(obs_next)
                        act_n_train.append(act)

                    embedding_loss_ag = embedding_model_ag.train_model(obs_n_train,obs_next_n_train,act_n_train)

            # save model, display training output
            if (terminal) and (len(episode_rewards) % arglist.save_rate == 0):
                U.save_state(arglist.save_dir, saver=saver)
                # print statement depends on whether or not there are adversaries
                if num_adversaries == 0:
                    print("steps: {}, episodes: {}, mean episode reward: {}, time: {}".format(
                        train_step, len(episode_rewards)-1, np.mean(episode_original_rewards[-arglist.save_rate-1:-1]), round(time.time()-t_start, 3)))
                else:
                    print("steps: {}, episodes: {}, mean episode reward: {}, agent episode reward: {}, time: {}".format(
                        train_step, len(episode_rewards)-1, np.mean(episode_original_rewards[-arglist.save_rate-1:-1]),
                        [np.mean(rew[-arglist.save_rate-1:-1]) for rew in agent_original_rewards], round(time.time()-t_start, 3)))
                
                # if arglist.reward_shaping_adv:
                #     print("adv agent original episode reward: {}".format(
                #         [np.mean(rew[-arglist.save_rate:]) for rew in agent_original_rewards[0:num_adversaries]]
                #     ))

                # if arglist.reward_shaping_ag:
                #     print("agent original episode reward: {}".format(
                #         [np.mean(rew[-arglist.save_rate:]) for rew in agent_original_rewards[num_adversaries:env.n]]
                #     ))
                t_start = time.time()
                # Keep track of final episode reward
                final_ep_rewards.append(np.mean(episode_original_rewards[-arglist.save_rate-1:-1]))
                for rew in agent_original_rewards:
                    final_ep_ag_rewards.append(np.mean(rew[-arglist.save_rate-1:-1]))

            # saves final episode reward for plotting training curve later
            if len(episode_rewards) > arglist.num_episodes:
                rew_file_name = arglist.plots_dir + arglist.exp_name + '_rewards.pkl'
                with open(rew_file_name, 'wb') as fp:
                    pickle.dump(final_ep_rewards, fp)
                agrew_file_name = arglist.plots_dir + arglist.exp_name + '_agrewards.pkl'
                with open(agrew_file_name, 'wb') as fp:
                    pickle.dump(final_ep_ag_rewards, fp)
                print('...Finished total of {} episodes.'.format(len(episode_rewards)))
                break

if __name__ == '__main__':
    arglist = parse_args()
    train(arglist)
