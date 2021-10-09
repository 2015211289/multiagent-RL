#!/bin/bash

# competitive_envs=("simple_adversary" "simple_crypto" "simple_push" "simple_tag" "simple_world_comm")
# coorperate_envs=("simple_reference" "simple_speaker_listener" "simple_spread")

# for env in ${coorperate_envs[*]}
# do
#     python ./maddpg_impl/experiments/train.py --scenario ${env} --reward-shaping-ag \
#     --exp-name "${env}1"
#     python ./maddpg_impl/experiments/train.py --scenario ${env} --good-policy maddpg \
#     --exp-name "${env}2" 
# done

# for env in ${competitive_envs[*]}
# do
#     python ./maddpg_impl/experiments/train.py --scenario ${env} --good-policy maddpg  \
#     --adv-policy maddpg --exp-name "${env}2" 
#     python ./maddpg_impl/experiments/train.py --scenario ${env} --adv-policy maddpg  \
#     --reward-shaping-ag --exp-name "${env}3"
#     python ./maddpg_impl/experiments/train.py --scenario ${env} --good-policy maddpg  \
#     --reward-shaping-adv --exp-name "${env}4"
# done

# 1: TD3
# 2: maddpg
# 3: good->TD3;adv->maddpg
# 4: good->maddpg;adv->TD3


envs=("Pong" "Box" "Tennis" "Basketball_Pong" "Double_Dunk" "Space_Invaders")
for((i=0;i<${#envs[@]};i++));do
    for((j=0;j<5;j++));do
        python ./maddpg_impl/experiments/train.py --scenario ${envs[$i]} --reward-shaping-ag \
        --exp-name $j --plots-dir "./20211008-1/RSMATD3/${envs[$i]}/" --pettingzoo --adv-policy maddpg
    done
done