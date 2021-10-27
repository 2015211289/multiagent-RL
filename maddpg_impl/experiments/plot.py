import pickle
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st

com_scene=["Tennis","Pong","Box","Double_Dunk","Wizard_of_Wor","Joust"]
coop_scene=["Mario_Bros","Space_Invaders"]
path = "./20211021/RSMATD3/"
N=100
M=30

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

'''获取各自奖励'''
def getAgRewards(scenario,i):

    with open(path+scenario+"/"+str(i)+"_agrewards.pkl", 'rb') as fp:
        agrewards = pickle.load(fp)

    # 计算每10轮的均值
    y = [[],[]]
    for i in range(0,2*N,2):
        sum1 = 0
        sum2 = 0
        count = 0

        for j in range(i-2*(M-1),i+1,2):
            if j >= 0:
                sum1 += agrewards[j]
                sum2 += agrewards[j+1]
                count +=1
            
        y[0].append(sum1/count)
        y[1].append(sum2/count)
        # y[0].append(agrewards[i])
        # y[1].append(agrewards[i+1])

    
    return y


'''获取合作奖励'''
def getEpRewards(scenario,i):

    with open(path+scenario+"/"+str(i)+"_rewards.pkl", 'rb') as fp:
        rewards = pickle.load(fp)
    
    with open(path+scenario+"/"+str(i)+"\'"+"_rewards.pkl", 'rb') as fp:  
        o_rewards = pickle.load(fp)

    # 计算每10轮的均值
    y = [[],[]]
    for i in range(0,N):
        sum1 = 0
        sum2 = 0
        count = 0

        for j in range(i-(M-1),i+1):
            if j >= 0:
                sum1 += rewards[j]
                sum2 += o_rewards[j]
                count +=1
            
        y[0].append(sum1/count)
        y[1].append(sum2/count)
        # y[0].append(rewards[i])
        # y[1].append(o_rewards[i])
    
    return y

if __name__ == '__main__':
    for env in com_scene:
        rewards=[]
        for i in range(10):
            rewards.append(getAgRewards(env,i))
        
        # 计算平均值和95%置信区间
        ave_adv_rewards=[]
        ave_ag_rewards=[]

        top_adv_rewards=[]
        bottom_adv_rewards=[]

        top_ag_rewards=[]
        bottom_ag_rewards=[]

        for i in range(N):
            tmp=[]
            tmp2=[]
            for j in range(10):
                tmp.append(rewards[j][0][i])
                tmp2.append(rewards[j][1][i])
            
            ave_adv_rewards.append(np.mean(tmp))
            ave_ag_rewards.append(np.mean(tmp2))

            (t,b)=st.t.interval(0.95,len(tmp)-1,loc=np.mean(tmp),scale=st.sem(tmp))
            top_adv_rewards.append(t)
            bottom_adv_rewards.append(b)

            (t,b)=st.t.interval(0.95,len(tmp2)-1,loc=np.mean(tmp2),scale=st.sem(tmp2))
            top_ag_rewards.append(t)
            bottom_ag_rewards.append(b)

        x = [i for i in range(1,N+1)]
        plt.plot(x,ave_adv_rewards,label="Exploration MATD3",color="red",linewidth=0.8)
        plt.plot(x,ave_ag_rewards,label="MATD3",color="blue",linewidth=0.8)
        plt.fill_between(x,bottom_adv_rewards,top_adv_rewards,color='red',alpha=0.1,linewidth=1)
        plt.fill_between(x,bottom_ag_rewards,top_ag_rewards,color='blue',alpha=0.1,linewidth=0.1)
        plt.xlabel('Episode')
        plt.ylabel('Agent Reward')
        plt.grid()
        plt.legend()
        # plt.savefig(env+".jpg")
        plt.show()
        plt.cla()
        print(env)
        print("adv reward: {}".format(ave_adv_rewards[-1]))
        print("ag reward: {}".format(ave_ag_rewards[-1]))


    for env in coop_scene:
        rewards=[]
        for i in range(10):
            rewards.append(getEpRewards(env,i))
        
        # 计算平均值和95%置信区间
        ave_ex_rewards=[]
        ave_rewards=[]

        top_ex_rewards=[]
        bottom_ex_rewards=[]

        top_rewards=[]
        bottom_rewards=[]

        for i in range(N):
            tmp=[]
            tmp2=[]
            for j in range(10):
                tmp.append(rewards[j][0][i])
                tmp2.append(rewards[j][1][i])
            
            ave_ex_rewards.append(np.mean(tmp))
            ave_rewards.append(np.mean(tmp2))

            (t,b)=st.t.interval(0.95,len(tmp)-1,loc=np.mean(tmp),scale=st.sem(tmp))
            top_ex_rewards.append(t)
            bottom_ex_rewards.append(b)

            (t,b)=st.t.interval(0.95,len(tmp2)-1,loc=np.mean(tmp2),scale=st.sem(tmp2))
            top_rewards.append(t)
            bottom_rewards.append(b)

        x = [i for i in range(1,N+1)]
        plt.plot(x,ave_ex_rewards,label="Exploration MATD3",color="red",linewidth=0.8)
        plt.plot(x,ave_rewards,label="MATD3",color="blue",linewidth=0.8)
        plt.fill_between(x,bottom_ex_rewards,top_ex_rewards,color='red',alpha=0.1,linewidth=0.1)
        plt.fill_between(x,bottom_rewards,top_rewards,color='blue',alpha=0.1,linewidth=0.1)
        plt.xlabel('Episode')
        plt.ylabel('Agent Reward')
        plt.grid()
        plt.legend()
        # plt.savefig(env+".jpg")    
        plt.show()
        plt.cla()
        print(env)
        print("ex reward: {}".format(ave_ex_rewards[-1]))
        print("reward: {}".format(ave_rewards[-1]))











    