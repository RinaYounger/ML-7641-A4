import numpy as np
import gym
import random
import time
import matplotlib.pyplot as plt
import gym
from gym import utils as utils
from gym.envs.toy_text.frozen_lake import generate_random_map
import NCPolicy as policy
import NCValue as value
import NCQLearning as Q
from hiivemdp import hiive
from hiive import mdptoolbox
import hiive.mdptoolbox.example
import hiive.mdptoolbox.mdp





def run(P, R, policy):
    # print("new")

    tot_rew = 0
    states = np.arange(len(policy))
    # print(policy)
    for x in range(100):
        # print("new round")
        state = 0
        go = True
        total = 0
        while go:

            action = policy[state]
            # print("action" + str(action))
            rew = R[state][action]
            # print("reward" + str(rew))
            total += rew
            probs = P[action][state]

            state = np.random.choice(states, 1, p=probs)[0]
            # print("new state" + str(state))
            if state == 0:
                go = False
        tot_rew+= total

    return tot_rew/100.0




if __name__ == '__main__':

    random.seed(26)
    utils.seeding.np_random(26)
    utils.seeding.hash_seed(26)
    utils.seeding.create_seed(26)
    np.random.seed(26)


    P, R = mdptoolbox.example.forest(S=500, r1 = 1000)



    gammas = [.1,.3,.5,.8,.9,.99]
    labels = [".1",'.3','.5','.8','.9','.99']
    pp = []

    for g in gammas:

        vi = mdptoolbox.mdp.PolicyIteration(P, R, gamma=g, max_iter=5000)
        vi.run()
        iters = len(vi.run_stats)
        rew = []
        for x in range (1, iters+1):

            vi = mdptoolbox.mdp.PolicyIteration(P, R, gamma=g, max_iter=x)
            vi.run()
            rew.append(run(P,R,vi.policy))
        pp.append(rew[-1])
        # print(np.mean(rew))
        # plt.plot(range(1, iters+1), rew, label = "Gamma = " + str(g))


    # plt.title("Forest Policy Iteration Reward vs Gamma")
    # plt.xlabel("Iterations")
    # plt.ylabel("Average Score on 100 runs")
    # plt.tight_layout()
    # plt.legend()
    # plt.savefig('F Policy vs Gamma Rewards.png')
    # plt.close()
    # plt.figure()

    plt.figure(figsize=(9, 5))
    plt.bar(labels, pp)
    plt.title("POLICY Reward vs Gamma")
    plt.xlabel("Gammas")
    plt.ylabel("Reward")
    plt.savefig('policy r vs gamma bar chart.png')
    plt.close()
    plt.figure()

    vp = []
    for g in gammas:

        vi = mdptoolbox.mdp.ValueIteration(P, R, gamma=g, max_iter=10000)
        vi.run()
        iters = len(vi.run_stats)
        rew = []

        for x in range (1, iters+1):

            vi = mdptoolbox.mdp.PolicyIteration(P, R, gamma=g, max_iter=x)
            vi.run()
            rew.append(run(P,R,vi.policy))
        vp.append(rew[-1])
        # print(np.mean(rew))
        # plt.plot(range(1, iters+1), rew, label = "Gamma = " + str(g))


    # plt.title("Forest Value Iteration Reward vs Gamma")
    # plt.xlabel("Iterations")
    # plt.ylabel("Average Score on 100 runs")
    # plt.tight_layout()
    # plt.legend()
    # plt.savefig('F Value vs Gamma Rewards.png')
    # plt.close()
    # plt.figure()

    plt.figure(figsize=(9, 5))
    plt.bar(labels, vp)
    plt.title("VALUE Reward vs Gamma")
    plt.xlabel("Gammas")
    plt.ylabel("Reward")
    plt.savefig('value r vs gamma bar chart.png')
    plt.close()
    plt.figure()



    #Gamma Stuff
    print("Policy")
    for x in gammas:
        print("gamma = " + str(x))
        rew = []
        times = []
        vi = mdptoolbox.mdp.PolicyIteration(P, R, gamma = x)
        vi.setVerbose()
        vi.run()
        for item in vi.run_stats:
            rew.append(item['Reward'])
            times.append(item['Time'])
        print(vi.policy)
        times = np.cumsum(times)

        plt.plot(range(1, len(rew) + 1), rew,
                     label="Gamma = " + str(x))

    plt.title("Forest Policy Iteration - Reward vs Gamma")
    plt.xlabel("Iterations")
    plt.ylabel("Reward")
    plt.tight_layout()
    plt.legend()
    plt.savefig('Forest POLICY Discounted Reward vs Gamma.png')
    plt.close()
    plt.figure()

    print("Value")
    for x in gammas:
        print("gamma = " + str(x))
        rew = []
        times = []
        vi = mdptoolbox.mdp.ValueIteration(P, R, gamma=x)
        vi.setVerbose()
        vi.run()
        for item in vi.run_stats:
            rew.append(item['Reward'])
            times.append(item['Time'])
        print(vi.policy)
        times = np.cumsum(times)
        plt.plot(range(1, len(rew) + 1), rew,
                 label="Gamma = " + str(x))

    plt.title("Forest Value Iteration - Reward vs Gamma")
    plt.xlabel("Iterations")
    plt.ylabel("Reward")
    plt.tight_layout()
    plt.legend()
    plt.savefig('Forest VALUE Discounted Reward vs Gamma.png')
    plt.close()
    plt.figure()


    vi = mdptoolbox.mdp.PolicyIteration(P, R, gamma=.9, max_iter=5000)
    vi.setVerbose()
    vi.run()
    print(vi.policy)

    vi = mdptoolbox.mdp.ValueIteration(P, R, gamma=.9, max_iter=5000)
    vi.setVerbose()
    vi.run()
    print(vi.policy)

    print("Q-learning")
    vi = mdptoolbox.mdp.QLearning(P, R, 0.9, n_iter = 100000000)
    vi.setVerbose()
    vi.run()
    print(vi.policy)
    print(vi.run_stats)



    vi = mdptoolbox.mdp.PolicyIteration(P, R, gamma=.99)
    # vi.setVerbose()
    vi.run()
    Ptimes = []
    for item in vi.run_stats:
        Ptimes.append(item['Time'])
    print(vi.policy)
    Ptimes = np.cumsum(Ptimes)

    vi = mdptoolbox.mdp.ValueIteration(P, R, gamma=.99)
    # vi.setVerbose()
    vi.run()
    Vtimes = []
    for item in vi.run_stats:
        Vtimes.append(item['Time'])
    print(vi.policy)
    Vtimes = np.cumsum(Vtimes)


    plt.plot(range(1, len(Ptimes) + 1), Ptimes,label="Policy Iteration")
    plt.plot(range(1, len(Vtimes) + 1), Vtimes, label="Value Iteration")

    plt.title("Forest Algo vs Time")
    plt.xlabel("Iterations")
    plt.ylabel("Time")
    plt.tight_layout()
    plt.legend()
    plt.savefig('Forest Algos vs Time.png')
    plt.close()
    plt.figure()

    vi = mdptoolbox.mdp.PolicyIteration(P, R, gamma=.99)
    # vi.setVerbose()
    vi.run()
    Ptimes = []
    for item in vi.run_stats:
        Ptimes.append(item['Reward'])
    print(vi.policy)
    # Ptimes = np.cumsum(Ptimes)

    vi = mdptoolbox.mdp.ValueIteration(P, R, gamma=.99)
    # vi.setVerbose()
    vi.run()
    Vtimes = []
    for item in vi.run_stats:
        Vtimes.append(item['Reward'])
    print(vi.policy)
    # Vtimes = np.cumsum(Vtimes)

    plt.plot(range(1, len(Ptimes) + 1), Ptimes, label="Policy Iteration")
    plt.plot(range(1, len(Vtimes) + 1), Vtimes, label="Value Iteration")

    plt.title("Forest Algo vs Discounted Reward")
    plt.xlabel("Iterations")
    plt.ylabel("Reward")
    plt.tight_layout()
    plt.legend()
    plt.savefig('Forest Algos vs Discounted Reward.png')
    plt.close()
    plt.figure()

    vi = mdptoolbox.mdp.PolicyIteration(P, R, gamma=.99)
    vi.run()
    Ptimes = []
    Pmaxv = []
    Pmeanv = []
    for item in vi.run_stats:
        Ptimes.append(item['Error'])
        Pmaxv.append(item['Max V'])
        Pmeanv.append(item['Mean V'])

    print(vi.policy)
    # Ptimes = np.cumsum(Ptimes)

    vi = mdptoolbox.mdp.ValueIteration(P, R, gamma=.99)
    # vi.setVerbose()
    vi.run()
    Vtimes = []
    Vmaxv = []
    Vmeanv = []
    for item in vi.run_stats:
        Vtimes.append(item['Error'])
        Vmaxv.append(item['Max V'])
        Vmeanv.append(item['Mean V'])
    print(vi.policy)
    # Vtimes = np.cumsum(Vtimes)

    plt.plot(range(1, len(Ptimes) + 1), Ptimes, label="Policy Iteration")
    plt.plot(range(1, len(Vtimes) + 1), Vtimes, label="Value Iteration")

    plt.title("Forest Algo vs Error")
    plt.xlabel("Iterations")
    plt.ylabel("Error")
    plt.tight_layout()
    plt.legend()
    plt.savefig('Forest Algos vs Error.png')
    plt.close()
    plt.figure()

    plt.plot(range(1, len(Pmaxv) + 1), Pmaxv, label="Policy Iteration")
    plt.plot(range(1, len(Vmaxv) + 1), Vmaxv, label="Value Iteration")

    plt.title("Forest Algo vs Max Value")
    plt.xlabel("Iterations")
    plt.ylabel("Max Value")
    plt.tight_layout()
    plt.legend()
    plt.savefig('Forest Algos vs Max Value.png')
    plt.close()
    plt.figure()

    plt.plot(range(1, len(Pmeanv) + 1), Pmeanv, label="Policy Iteration")
    plt.plot(range(1, len(Vmeanv) + 1), Vmeanv, label="Value Iteration")

    plt.title("Forest Algo vs Mean Value")
    plt.xlabel("Iterations")
    plt.ylabel("Mean Value")
    plt.tight_layout()
    plt.legend()
    plt.savefig('Forest Algos vs Mean Value.png')
    plt.close()
    plt.figure()





    vi = mdptoolbox.mdp.PolicyIteration(P, R, gamma=.99)
    vi.run()
    iters = len(vi.run_stats)
    Prew = []
    for x in range (1, iters+1):

        vi = mdptoolbox.mdp.PolicyIteration(P, R, gamma=.99, max_iter=x)
        vi.run()
        Prew.append(run(P,R,vi.policy))

    print("POLICY " + str(run(P,R,vi.policy)))

    vi = mdptoolbox.mdp.ValueIteration(P, R, gamma=.99)
    vi.run()
    iters = len(vi.run_stats)
    Vrew = []
    for x in range (1, iters+1):

        vi = mdptoolbox.mdp.ValueIteration(P, R, gamma=.99, max_iter=x)
        vi.run()
        Vrew.append(run(P,R,vi.policy))

    print("VALUE " + str(run(P, R, vi.policy)))

    plt.plot(range(1, len(Prew) + 1), Prew, label="Policy Iteration")
    plt.plot(range(1, len(Vrew) + 1), Vrew, label="Value Iteration")

    plt.title("Forest Algo vs Reward")
    plt.xlabel("Iterations")
    plt.ylabel("Reward")
    plt.tight_layout()
    plt.legend()
    plt.savefig('Forest Algos vs Relative Reward.png')
    plt.close()
    plt.figure()



    #Sizes
    sizes = [10,50,500]
    rw = [1000,1000,1000]

    policy_rewards= []
    value_rewards = []
    pd = []
    vd = []
    pm = []
    vm = []

    for x in range(len(sizes)):

        print('size' + str(sizes[x]))

        P, R = mdptoolbox.example.forest(S=sizes[x], r1 = rw[x])

        vi = mdptoolbox.mdp.PolicyIteration(P, R, gamma=.99)
        vi.run()
        print("Policy")
        print(vi.policy)

        disc = []
        misc = []
        for item in vi.run_stats:
            disc.append(item['Max V'])
            misc.append(item["Mean V"])

        pd.append(disc)
        pm.append(misc)

        iters = len(vi.run_stats)
        Prew = []
        for x in range (1, iters+1):

            vi = mdptoolbox.mdp.PolicyIteration(P, R, gamma=.99, max_iter=x)
            vi.run()
            Prew.append(run(P,R,vi.policy))

        policy_rewards.append(Prew)


        vi = mdptoolbox.mdp.ValueIteration(P, R, gamma=.99)
        vi.run()
        print("Policy")
        print(vi.policy)
        disc = []
        misc = []
        for item in vi.run_stats:
            disc.append(item['Max V'])
            misc.append(item["Mean V"])
        vd.append(disc)
        vm.append(misc)
        iters = len(vi.run_stats)
        Vrew = []
        for x in range (1, iters+1):

            vi = mdptoolbox.mdp.ValueIteration(P, R, gamma=.99, max_iter=x)
            vi.run()
            Vrew.append(run(P,R,vi.policy))

        value_rewards.append(Vrew)


    # for x in range(len(sizes)):
    #     plt.plot(range(1, len(policy_rewards[x]) + 1), policy_rewards[x], label="Size = " + str(sizes[x]))
    # plt.title("Forest Policy Reward vs Problem Size")
    # plt.xlabel("Iterations")
    # plt.ylabel("Average Score on 100 runs")
    # plt.ylim((0,2))
    # plt.tight_layout()
    # plt.legend()
    # plt.savefig('Forest POLICY vs Size Rewards zoomed in.png')
    # plt.close()
    # plt.figure()
    #
    # for x in range(len(sizes)):
    #     plt.plot(range(1, len(value_rewards[x]) + 1), value_rewards[x], label="Size = " + str(sizes[x]))
    # plt.title("Forest Value Reward vs Problem Size")
    # plt.xlabel("Iterations")
    # plt.ylabel("Average Score on 100 runs")
    # plt.ylim((0,2))
    # plt.tight_layout()
    # plt.legend()
    # plt.savefig('Forest VALUE vs Size Rewards zoomed in.png')
    # plt.close()
    # plt.figure()
    #
    #
    # for x in range(len(sizes)):
    #     plt.plot(range(1, len(policy_rewards[x]) + 1), policy_rewards[x], label="Size = " + str(sizes[x]))
    # plt.title("Forest Policy Reward vs Problem Size")
    # plt.xlabel("Iterations")
    # plt.ylabel("Average Score on 100 runs")
    # plt.tight_layout()
    # plt.legend()
    # plt.savefig('Forest POLICY vs Size Rewards zoomed out.png')
    # plt.close()
    # plt.figure()
    #
    # for x in range(len(sizes)):
    #     plt.plot(range(1, len(value_rewards[x]) + 1), value_rewards[x], label="Size = " + str(sizes[x]))
    # plt.title("Forest Value Reward vs Problem Size")
    # plt.xlabel("Iterations")
    # plt.ylabel("Average Score on 100 runs")
    # plt.tight_layout()
    # plt.legend()
    # plt.savefig('Forest VALUE vs Size Rewards zoomed out.png')
    # plt.close()
    # plt.figure()

    # for x in range(len(sizes)):
    #     plt.plot(range(1, len(pd[x]) + 1), pd[x], label="Size = " + str(sizes[x]))
    # plt.title("Forest Policy Maximum Utility vs Problem Size")
    # plt.xlabel("Iterations")
    # plt.ylabel("Utility")
    # plt.tight_layout()
    # plt.legend()
    # plt.savefig('Forest POLICY vs Size Max V.png')
    # plt.close()
    # plt.figure()
    #
    # for x in range(len(sizes)):
    #     plt.plot(range(1, len(vd[x]) + 1), vd[x],
    #              label="Size = " + str(sizes[x]))
    # plt.title("Forest Value Maximum Utility vs Problem Size")
    # plt.xlabel("Iterations")
    # plt.ylabel("Utility")
    # plt.tight_layout()
    # plt.legend()
    # plt.savefig('Forest VALUE vs Size Max V.png')
    # plt.close()
    # plt.figure()

    for x in range(len(sizes)):
        plt.plot(range(1, len(pm[x]) + 1), pm[x], label="Size = " + str(sizes[x]))
    plt.title("Forest Policy Mean Utility vs Problem Size")
    plt.xlabel("Iterations")
    plt.ylabel("Utility")
    plt.tight_layout()
    plt.legend()
    plt.savefig('Forest POLICY vs Size Mean V.png')
    plt.close()
    plt.figure()

    for x in range(len(sizes)):
        plt.plot(range(1, len(vm[x]) + 1), vm[x],
                 label="Size = " + str(sizes[x]))
    plt.title("Forest Value Mean Utility vs Problem Size")
    plt.xlabel("Iterations")
    plt.ylabel("Utility")
    plt.tight_layout()
    plt.legend()
    plt.savefig('Forest VALUE vs Size Mean V.png')
    plt.close()
    plt.figure()

    random.seed(42)
    np.random.seed(42)




    print("Q-learning")
    vi = mdptoolbox.mdp.QLearning(P, R, 0.99, epsilon_decay=.001, epsilon = 1,  n_iter = 100000000)# to converge in 100 million use .9 gamma instead
    vi.setVerbose()
    vi.run()
    print(vi.policy)
    # print(vi.run_stats)

    drew = []
    times = []
    errors = []
    meanv = []
    maxv = []

    for item in vi.run_stats:
        drew.append(item['Reward'])
        times.append(item['Time'])
        errors.append(item['Error'])
        meanv.append(item['Mean V'])
        maxv.append(item['Max V'])


    times = np.cumsum(times)

    plt.plot(range(1, len(drew) + 1), drew, label="Q Learning")
    plt.title("Forest QL Iteration vs Discounted Reward")
    plt.xlabel("Iterations")
    plt.ylabel("Discounted Reward")
    plt.tight_layout()
    plt.legend()
    plt.savefig('Forest QL vs Discounted Reward final.png')
    plt.close()
    plt.figure()

    plt.plot(range(1, len(times) + 1), times, label="Q Learning")
    plt.title("Forest QL Iteration vs time")
    plt.xlabel("Iterations")
    plt.ylabel("Times")
    plt.tight_layout()
    plt.legend()
    plt.savefig('Forest QL vs Time final.png')
    plt.close()
    plt.figure()

    plt.plot(range(1, len(errors) + 1), errors, label="Q Learning")
    plt.title("Forest QL Iteration vs Error")
    plt.xlabel("Iterations")
    plt.ylabel("Error")
    plt.tight_layout()
    plt.legend()
    plt.savefig('Forest QL vs Error final.png')
    plt.close()
    plt.figure()

    plt.plot(range(1, len(meanv) + 1), meanv, label="Q Learning")
    plt.title("Forest QL Iteration vs Mean V")
    plt.xlabel("Iterations")
    plt.ylabel("Mean V")
    plt.tight_layout()
    plt.legend()
    plt.savefig('Forest QL vs Mean V final.png')
    plt.close()
    plt.figure()

    plt.plot(range(1, len(maxv) + 1), maxv, label="Q Learning")
    plt.title("Forest QL Iteration vs Max V")
    plt.xlabel("Iterations")
    plt.ylabel("Max V")
    plt.tight_layout()
    plt.legend()
    plt.savefig('Forest QL vs Max V final.png')
    plt.close()
    plt.figure()


    Ptimes = []
    Vtimes = []
    Qtimes = []

    Pmax = []
    Vmax = []
    Qmax = []

    Pmean = []
    Vmean = []
    Qmean = []

    vi = mdptoolbox.mdp.PolicyIteration(P, R, gamma=.99)
    vi.run()
    for item in vi.run_stats:
        Ptimes.append(item['Time'])
        Pmean.append(item['Mean V'])
        Pmax.append(item['Max V'])
    print(vi.policy)
    Ptimes = np.cumsum(Ptimes)

    vi = mdptoolbox.mdp.ValueIteration(P, R, gamma=.99)
    vi.run()
    for item in vi.run_stats:
        Vtimes.append(item['Time'])
        Vmean.append(item['Mean V'])
        Vmax.append(item['Max V'])
    print(vi.policy)
    Vtimes = np.cumsum(Vtimes)

    vi = mdptoolbox.mdp.QLearning(P, R, gamma=.99, epsilon=1, epsilon_decay=.001, n_iter=100000000)
    vi.run()
    for item in vi.run_stats:
        Qtimes.append(item['Time'])
        Qmean.append(item['Mean V'])
        Qmax.append(item['Max V'])
    print(vi.policy)
    Qtimes = np.cumsum(Qtimes)

    plt.plot(range(1, len(Ptimes) + 1), np.log10(Ptimes), label="Policy Iteration")
    plt.plot(range(1, len(Vtimes) + 1), np.log10(Vtimes), label="Value Iteration")
    plt.plot(range(1, len(Qtimes) + 1), np.log10(Qtimes), label="Q Learning")

    plt.title("Forest Algorithms vs Logarithmic Time")
    plt.xlabel("Iterations")
    plt.ylabel("Time")
    plt.tight_layout()
    plt.legend()
    plt.savefig('Forest All Algos vs Time new.png')
    plt.close()
    plt.figure()

    plt.plot(range(1, len(Pmax) + 1), Pmax, label="Policy Iteration")
    plt.plot(range(1, len(Vmax) + 1), Vmax, label="Value Iteration")
    plt.plot(range(1, len(Qmax) + 1), Qmax, label="Q Learning")

    plt.title("Forest Algorithms vs Maximum Utility")
    plt.xlabel("Iterations")
    plt.ylabel("Utility")
    plt.tight_layout()
    plt.legend()
    plt.savefig('Forest All Algos vs Max V new.png')
    plt.close()
    plt.figure()

    plt.plot(range(1, len(Pmean) + 1), Pmean, label="Policy Iteration")
    plt.plot(range(1, len(Vmean) + 1), Vmean, label="Value Iteration")
    plt.plot(range(1, len(Qmean) + 1), Qmean, label="Q Learning")

    plt.title("Forest Algorithms vs Mean V")
    plt.xlabel("Iterations")
    plt.ylabel("Utility")
    plt.tight_layout()
    plt.legend()
    plt.savefig('Forest All Algos vs Mean V new.png')
    plt.close()
    plt.figure()

    epsilon = [.1,.3,.5,.8,.9,.99]
    print("Q Gamma")
    errors = []
    t = []
    maxv = []
    meanv = []
    for x in epsilon:
        rew = []
        times = []
        error = []
        mav = []
        mev = []
        vi = mdptoolbox.mdp.QLearning(P, R, gamma = x, epsilon=1, n_iter=100000000)
        vi.setVerbose()
        vi.run()
        for item in vi.run_stats:
            rew.append(item['Reward'])
            times.append(item['Time'])
            error.append(item['Error'])
            mav.append(item['Max V'])
            mev.append(item['Mean V'])

        print("gamma = " + str(x))
        print(vi.policy)
        times = np.cumsum(times)
        errors.append(error)
        t.append(times)
        maxv.append(mav)
        meanv.append(mev)

        plt.plot(range(1, len(rew) + 1), rew,
                 label="gamma = " + str(x))

    plt.title("Forest QLearning - Reward vs Gamma")
    plt.xlabel("Iterations")
    plt.ylabel("Reward")
    plt.tight_layout()
    plt.legend()
    plt.savefig('Forest QL Reward vs Gamma.png')
    plt.close()
    plt.figure()

    for x in range(len(epsilon)):
        plt.plot(range(1, len(errors[x]) + 1), errors[x],
                 label="gamma = " + str(epsilon[x]))

    plt.title("Forest QLearning - Error vs Gamma")
    plt.xlabel("Iterations")
    plt.ylabel("Error")
    plt.tight_layout()
    plt.legend()
    plt.savefig('Forest QL Error vs Gamma.png')
    plt.close()
    plt.figure()

    for x in range(len(epsilon)):
        plt.plot(range(1, len(t[x]) + 1), t[x],
                 label="gamma = " + str(epsilon[x]))

    plt.title("Forest QLearning - Time vs Gamma")
    plt.xlabel("Iterations")
    plt.ylabel("Time")
    plt.tight_layout()
    plt.legend()
    plt.savefig('Forest QL Time vs Gamma.png')
    plt.close()
    plt.figure()

    for x in range(len(epsilon)):
        plt.plot(range(1, len(meanv[x]) + 1), meanv[x],
                 label="gamma = " + str(epsilon[x]))

    plt.title("Forest QLearning - Mean Value vs Gamma")
    plt.xlabel("Iterations")
    plt.ylabel("Mean Value")
    plt.tight_layout()
    plt.legend()
    plt.savefig('Forest QL Mean V vs Gamma.png')
    plt.close()
    plt.figure()

    for x in range(len(epsilon)):
        plt.plot(range(1, len(maxv[x]) + 1), maxv[x],
                 label="gamma = " + str(epsilon[x]))

    plt.title("Forest QLearning - Max Value vs Gamma")
    plt.xlabel("Iterations")
    plt.ylabel("Max Value")
    plt.tight_layout()
    plt.legend()
    plt.savefig('Forest QL Max V vs Gamma.png')
    plt.close()
    plt.figure()

    # epsilon = [.01, .001, .0001, .00001, .000001]
    # epsilon = [.99, .1, .01,.001, .0001]
    epsilon = [.01, .001, .0001, .00001, .000001]
    print("Q epsilon decay")
    errors = []
    t = []
    maxv = []
    meanv = []
    for x in epsilon:
        rew = []
        times = []
        error = []
        mav = []
        mev = []
        vi = mdptoolbox.mdp.QLearning(P, R, gamma=.99, epsilon=1,epsilon_decay=x, n_iter=100000000)
        vi.setVerbose()
        vi.run()
        for item in vi.run_stats:
            rew.append(item['Reward'])
            times.append(item['Time'])
            error.append(item['Error'])
            mav.append(item['Max V'])
            mev.append(item['Mean V'])

        print("epsilon decay = " + str(x))
        print(vi.policy)
        times = np.cumsum(times)
        errors.append(error)
        t.append(times)
        maxv.append(mav)
        meanv.append(mev)

        plt.plot(range(1, len(rew) + 1), rew,
                 label="epsilon decay= " + str(x))

    plt.title("Forest QLearning - Reward vs Epsilon Decay")
    plt.xlabel("Iterations")
    plt.ylabel("Reward")
    plt.tight_layout()
    plt.legend()
    plt.savefig('Forest QL Reward vs Epsilon Decay xx.png')
    plt.close()
    plt.figure()

    for x in range(len(epsilon)):
        plt.plot(range(1, len(errors[x]) + 1), errors[x],
                 label="Epsilon Decay = " + str(epsilon[x]))

    plt.title("Forest QLearning - Error vs Epsilon Decay")
    plt.xlabel("Iterations")
    plt.ylabel("Error")
    plt.tight_layout()
    plt.legend()
    plt.savefig('Forest QL Error vs Epsilon Decay xx.png')
    plt.close()
    plt.figure()

    for x in range(len(epsilon)):
        plt.plot(range(1, len(t[x]) + 1), t[x],
                 label="Epsilon Decay = " + str(epsilon[x]))

    plt.title("Forest QLearning - Time vs Epsilon Decay")
    plt.xlabel("Iterations")
    plt.ylabel("Time")
    plt.tight_layout()
    plt.legend()
    plt.savefig('Forest QL Time vs Epsilon Decay xx.png')
    plt.close()
    plt.figure()

    for x in range(len(epsilon)):
        plt.plot(range(1, len(meanv[x]) + 1), meanv[x],
                 label="Epsilon Decay = " + str(epsilon[x]))

    plt.title("Forest QLearning - Mean Value vs Epsilon Decay")
    plt.xlabel("Iterations")
    plt.ylabel("Mean Value")
    plt.tight_layout()
    plt.legend()
    plt.savefig('Forest QL Mean V vs Epsilon Decay xx.png')
    plt.close()
    plt.figure()

    for x in range(len(epsilon)):
        plt.plot(range(1, len(maxv[x]) + 1), maxv[x],
                 label="Epsilon Decay = " + str(epsilon[x]))

    plt.title("Forest QLearning - Max Value vs Epsilon Decay")
    plt.xlabel("Iterations")
    plt.ylabel("Max Value")
    plt.tight_layout()
    plt.legend()
    plt.savefig('Forest QL Max V vs Epsilon Decay xx.png')
    plt.close()
    plt.figure()





    epsilon = [.1,.5,.8,1,1.5]
    print("Q")
    errors = []
    t = []
    maxv = []
    meanv = []
    for x in epsilon:
        rew = []
        times = []
        error = []
        mav = []
        mev = []
        vi = mdptoolbox.mdp.QLearning(P, R, .99, epsilon=x, n_iter=100000000)
        vi.setVerbose()
        vi.run()
        for item in vi.run_stats:
            rew.append(item['Reward'])
            times.append(item['Time'])
            error.append(item['Error'])
            mav.append(item['Max V'])
            mev.append(item['Mean V'])

        print("epsilon = " + str(x))
        print(vi.policy)
        times = np.cumsum(times)
        errors.append(error)
        t.append(times)
        maxv.append(mav)
        meanv.append(mev)

        plt.plot(range(1, len(rew) + 1), rew,
                     label="epsilon = " + str(x))

    plt.title("Forest QLearning - Reward vs Epsilon")
    plt.xlabel("Iterations")
    plt.ylabel("Reward")
    plt.tight_layout()
    plt.legend()
    plt.savefig('Forest QL Reward vs Epsilon x.png')
    plt.close()
    plt.figure()

    for x in range(len(epsilon)):
        plt.plot(range(1, len(errors[x]) + 1), errors[x],
             label="epsilon = " + str(epsilon[x]))

    plt.title("Forest QLearning - Error vs Epsilon")
    plt.xlabel("Iterations")
    plt.ylabel("Error")
    plt.tight_layout()
    plt.legend()
    plt.savefig('Forest QL Error vs Epsilon x.png')
    plt.close()
    plt.figure()

    for x in range(len(epsilon)):
        plt.plot(range(1, len(t[x]) + 1), t[x],
             label="epsilon = " + str(epsilon[x]))

    plt.title("Forest QLearning - Time vs Epsilon")
    plt.xlabel("Iterations")
    plt.ylabel("Time")
    plt.tight_layout()
    plt.legend()
    plt.savefig('Forest QL Time vs Epsilon x.png')
    plt.close()
    plt.figure()

    for x in range(len(epsilon)):
        plt.plot(range(1, len(meanv[x]) + 1), meanv[x],
             label="epsilon = " + str(epsilon[x]))

    plt.title("Forest QLearning - Mean Value vs Epsilon")
    plt.xlabel("Iterations")
    plt.ylabel("Mean Value")
    plt.tight_layout()
    plt.legend()
    plt.savefig('Forest QL Mean V vs Epsilon x.png')
    plt.close()
    plt.figure()

    for x in range(len(epsilon)):
        plt.plot(range(1, len(maxv[x]) + 1), maxv[x],
                 label="epsilon = " + str(epsilon[x]))

    plt.title("Forest QLearning - Max Value vs Epsilon")
    plt.xlabel("Iterations")
    plt.ylabel("Max Value")
    plt.tight_layout()
    plt.legend()
    plt.savefig('Forest QL Max V vs Epsilon x.png')
    plt.close()
    plt.figure()








