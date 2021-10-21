import pickle
import matplotlib.pyplot as plt
import numpy as np

com_scene=["Tennis" "Pong" "Boxing" "Double_Dunk"]
coop_scene=["Mario_Bros","Space_Invaders"]
path = "../../20211021/RSMATD3/"
N=100


# with open('./learning_curves/TD3(r)_agrewards.pkl', 'rb') as fp:
#     agent_rewards = pickle.load(fp)

# with open("./learning_curves/maddpgVSTD3rs_agrewards.pkl", 'rb') as fp:
#     td3rsVSmaddpg = pickle.load(fp)

# with open('./learning_curves/maddpg_agrewards.pkl', 'rb') as fp:
#     agent_rewards = pickle.load(fp)

# with open("./learning_curves/TD3rsVSmaddpg_agrewards.pkl", 'rb') as fp:
#     td3rsVSmaddpg = pickle.load(fp)


# x = [ i for i in range(len(td3rsVSmaddpg)//(advNum+goodNum))]
# plt.plot(td3rsVSmaddpg[0:len(td3rsVSmaddpg)-1:advNum+goodNum],label = "adv-MATD3RS1")
# plt.plot(td3rsVSmaddpg[1:len(td3rsVSmaddpg)-1:advNum+goodNum],label = "adv-MATD3RS2")
# plt.plot(td3rsVSmaddpg[2:len(td3rsVSmaddpg)-1:advNum+goodNum],label = "adv-MATD3RS3")
# plt.plot(td3rsVSmaddpg[3:len(td3rsVSmaddpg)-1:advNum+goodNum],label = "adv-MATD3RS4")
# plt.plot(td3rsVSmaddpg[4:len(td3rsVSmaddpg)-1:advNum+goodNum],label = "MADDPG1")
# plt.plot(td3rsVSmaddpg[5:len(td3rsVSmaddpg)-1:advNum+goodNum],label = "MADDPG2")

# # plt.plot(td3_rewards,label = "MATD3RS")
# plt.xticks(x)
# plt.grid()
# plt.xlabel('per 1000 episode')
# plt.ylabel('agent reward')
# plt.legend()
# plt.show()

def getAgRewards(scenario,i):
    with open(path+scenario+"/"+i+"_agrewards.pkl", 'rb') as fp:
        agrewards = pickle.load(fp)

    # 计算每10轮的均值
    y = [[],[]]
    for i in range(0,N,2):
        sum1 = 0
        sum2 = 0
        count = 0

        for j in range(i-20,i+1,2):
            if j >= 0:
                sum1 += agrewards[j]
                sum2 +=agrewards[j+1]
                count +=1
            
        y[0].append(sum1/count)
        y[1].append(sum2/count)
    
    return y


if __name__ == '__main__':
    for env in com_scene:
        rewards=[]
        for i in range(10):
            rewards.append(getAgRewards(env,i))
        
        # 计算平均值和95%置信区间
        adv_rewards=[]
        ag_rewards=[]
        
        for i in range(10):



    