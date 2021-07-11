#!/bin/bash

competitive_envs=("simple_adversary" "simple_crypto" "simple_push simple_tag" "simple_world_comm")
coorperate_envs=("simple_reference" "simple_speaker_listener" "simple_spread")

# TD3(r)
for env in ${coorperate_envs[*]}
do
    nohup python ./maddpg_impl/maddpg_impl/experiments/train.py --scenario ${env} --num-episodes 10000 --reward-shaping-ag \
    --exp-name "TD3(r)-${env}"
done
# TD3 
for env in ${coorperate_envs[*]}
do
    nohup python ./maddpg_impl/maddpg_impl/experiments/train.py --scenario ${env} --num-episodes 10000 \
    --exp-name "TD3-${env}" 
done

# maddpg
for env in ${coorperate_envs[*]}
do
    nohup python ./maddpg_impl/maddpg_impl/experiments/train.py --scenario ${env} --good-policy maddpg --num-episodes 10000 \
    --exp-name "maddpg-${env}" 
done

# TD3(r) compete with TD3(r)
for env in ${competitive_envs[*]}
do
    nohup python ./maddpg_impl/maddpg_impl/experiments/train.py --scenario ${env} --num-episodes 10000 --reward-shaping-ag --reward-shaping-adv \
    --exp-name "TD3(r)vsTD3(r)-${env}" 
done

# TD3(r) compete with maddpg
for env in ${competitive_envs[*]}
do
    nohup python ./maddpg_impl/maddpg_impl/experiments/train.py --scenario ${env} --num-episodes 10000 --reward-shaping-ag  --adv-policy maddpg \
    --exp-name "TD3(r)vsMADDPG-${env}" 
done

# maddpg compete with TD3(r)
for env in ${competitive_envs[*]}
do
    nohup python ./maddpg_impl/maddpg_impl/experiments/train.py --scenario ${env} --num-episodes 10000 --reward-shaping-adv  --good-policy maddpg \
    --exp-name "MADDPGvsTD3(r)-${env}" 
done