import pickle
import matplotlib.pyplot as plt

cooperationScene = ["simple_reference","simple_speaker_listener","simple_spread"]
advScene =["simple_adversary","simple_crypto","simple_push","simple_tag","simple_world_comm"]
advNum = [0,0,0,1,1,1,3,3]
goodNum = [2,2,3,2,2,1,1,2]
path = "./learning_curves/"
index = 0
for scene in cooperationScene:
    value = []*2
    for i in range(2):
        fileName = path+scene+i+"_agrewards.pkl"
        with open(fileName,"rb") as fp:
            value[i] = pickle.load(fp)


    maddpg = 0
    td3 = 0
    for i in range(goodNum[index]):
        maddpg+=value[0][-1-i]
        td3 +=value[1][-1-i]
    print("maddpg:"+str(maddpg))
    print("TD3:"+str(td3))

for()


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

def createPlot(scenario,i,adv_num,num):
    agrewards = []
    rewards = []
    with open("./learning_curves/"+scenario+i+"_agrewards.pkl", 'rb') as fp:
        agrewards = pickle.load(fp)

    with open("./learning_curves/"+scenario+i+"_rewards.pkl", 'rb') as fp:
        rewards = pickle.load(fp)

    x = [ i for i in range(len(rewards))]
    y = [[],[]]
    for i in range(0,len(agrewards),num):
        sum=0
        for j in range(adv_num):
            sum+=agrewards[i+j]
        y[0].append(sum)


    