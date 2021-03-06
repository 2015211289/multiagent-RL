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


# com_envs=("Tennis" "Othello" "Surround" "Video_Checkers")
# for((i=0;i<${#com_envs[@]};i++));do
#     for((j=0;j<1;j++));do
#         python ./maddpg_impl/experiments/train.py --scenario ${com_envs[$i]} --reward-shaping-adv --num-adversaries 1 \
#         --exp-name $j --plots-dir "./20211127/RSMATD3/${com_envs[$i]}/" --pettingzoo --save-dir "./20211127/RSMATD3/${com_envs[$i]}/model/"
#     done
# done

coop_envs=("Wizard_of_Wor" "Joust")
for((i=0;i<${#coop_envs[@]};i++));do
    for((j=0;j<1;j++));do
        python ./maddpg_impl/experiments/train.py --scenario ${coop_envs[$i]} --reward-shaping-adv --num-adversaries 1 \
        --exp-name $j --plots-dir "./20211206/RSMATD3/${coop_envs[$i]}/" --pettingzoo --save-dir "./20211206/RSMATD3/${coop_envs[$i]}/model/"
    done
done

# coop_envs=("Mario_Bros" "Space_Invaders")
# for((i=0;i<${#coop_envs[@]};i++));do
#     for((j=0;j<1;j++));do
#         python ./maddpg_impl/experiments/train.py --scenario ${coop_envs[$i]} --num-adversaries 0 \
#         --exp-name "$j'" --plots-dir "./20211125/RSMATD3/${coop_envs[$i]}/" --pettingzoo --save-dir "./202111225/RSMATD3/${coop_envs[$i]}/model'/"
#     done
# done

# com_envs=("Wizard_of_Wor" "Joust")
# for((i=0;i<${#com_envs[@]};i++));do
#     for((j=0;j<1;j++));do
#         python ./maddpg_impl/experiments/train.py --scenario ${com_envs[$i]} --reward-shaping-adv --num-adversaries 1 \
#         --exp-name $j --plots-dir "./20211127/RSMATD3/${com_envs[$i]}/" --pettingzoo --save-dir "./20211127/RSMATD3/${com_envs[$i]}/model/"
#     done
# done