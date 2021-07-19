import pickle
import matplotlib.pyplot as plt

advNum = 4
goodNum = 2
td3rsVSmaddpg = None

# with open('./learning_curves/TD3(r)_agrewards.pkl', 'rb') as fp:
#     agent_rewards = pickle.load(fp)

with open("./learning_curves/maddpgVSTD3rs_agrewards.pkl", 'rb') as fp:
    td3rsVSmaddpg = pickle.load(fp)

# with open('./learning_curves/maddpg_agrewards.pkl', 'rb') as fp:
#     agent_rewards = pickle.load(fp)

# with open("./learning_curves/TD3rsVSmaddpg_agrewards.pkl", 'rb') as fp:
#     td3rsVSmaddpg = pickle.load(fp)


x = [ i for i in range(len(td3rsVSmaddpg)//(advNum+goodNum))]
plt.plot(td3rsVSmaddpg[0:len(td3rsVSmaddpg)-1:advNum+goodNum],label = "adv-MATD3RS1")
plt.plot(td3rsVSmaddpg[1:len(td3rsVSmaddpg)-1:advNum+goodNum],label = "adv-MATD3RS2")
plt.plot(td3rsVSmaddpg[2:len(td3rsVSmaddpg)-1:advNum+goodNum],label = "adv-MATD3RS3")
plt.plot(td3rsVSmaddpg[3:len(td3rsVSmaddpg)-1:advNum+goodNum],label = "adv-MATD3RS4")
plt.plot(td3rsVSmaddpg[4:len(td3rsVSmaddpg)-1:advNum+goodNum],label = "MADDPG1")
plt.plot(td3rsVSmaddpg[5:len(td3rsVSmaddpg)-1:advNum+goodNum],label = "MADDPG2")

# plt.plot(td3_rewards,label = "MATD3RS")
plt.xticks(x)
plt.grid()
plt.xlabel('per 1000 episode')
plt.ylabel('agent reward')
plt.legend()
plt.show()