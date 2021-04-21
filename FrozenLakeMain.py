import numpy as np
import gym
import random
import time
import matplotlib.pyplot as plt
import gym
from gym import utils as utils
from gym.envs.toy_text.frozen_lake import generate_random_map
import FLPolicy as policy
import FLValue as value
import FLQLearning as Q
import plotGrid2 as pg2
import plotGrid as pg


#From https://learning.oreilly.com/library/view/reinforcement-learning-algorithms/9781789131116/7c6dfed0-1180-49fe-84a0-ea62131b5947.xhtml


def get_rewards(env, policy):


    tot_rew = 0
    state = env.reset()
    for _ in range(100):
        done = False
        while not done:
            # select the action accordingly to the policy

            next_state, reward, done, _ = env.step(policy[state])

            state = next_state
            tot_rew += reward
            if done:
                state = env.reset()
    return tot_rew


if __name__ == '__main__':

    random.seed(26)
    utils.seeding.np_random(26)
    utils.seeding.hash_seed(26)
    utils.seeding.create_seed(26)
    np.random.seed(26)

    env = gym.make('FrozenLake-v0')
    env.seed(613)

    # random.seed(26)
    # utils.seeding.np_random(26)
    # utils.seeding.hash_seed(26)
    # utils.seeding.create_seed(26)
    # np.random.seed(26)
    #
    #
    # ##Reg 4x4
    # env = gym.make('FrozenLake-v0')
    # env = env.unwrapped
    # env.seed(613)

    row, col = env.s // env.ncol, env.s % env.ncol
    desc = env.desc.tolist()
    desc = [[c.decode('utf-8') for c in line] for line in desc]
    pg2.main(desc)

    #
    # random.seed(26)
    # utils.seeding.np_random(26)
    # utils.seeding.hash_seed(26)
    # utils.seeding.create_seed(26)
    # np.random.seed(26)

    ps = []
    p, r = Q.main(env, 4, "4x4")
    ps.append(p)
    p = p.reshape((4, 4))
    print(p)
    pg.main(p, "FL QL 4x4 Policy Visualization.png")
    env.reset()


    p, r = policy.main(env, 4, "4x4",.99, True)
    print(p)
    ps.append(p)
    p = p.reshape((4, 4))
    pg.main(p, "FL POLICY ITER 4x4 Policy Visualization.png")
    env.reset()

    p, r = value.main(env, 4, "4x4", .99, True)
    print(p)
    ps.append(p)
    p = p.reshape((4, 4))
    print(p)
    pg.main(p, "FL Value ITER 4x4 Policy Visualization.png")
    env.reset()


    s = ["QL","Policy", "Value"]
    for x in range(len(s)):
        env.seed(26)
        tot_rew = 0
        state = env.reset()
        for _ in range(100):
            done = False
            while not done:
                # select the action accordingly to the policy
                next_state, reward, done, _ = env.step(ps[x][state])

                state = next_state
                tot_rew += reward
                if done:
                    state = env.reset()
        print(s[x])
        print('Won %i of %i games!' % (tot_rew, 100))


    gammas = [.1,.5,.8,.99]
    prs = []
    vrs = []

    for x in gammas:

        env.reset()
        policy_policy, policy_reward = policy.main(env, 4, "4x4 Gamma= " + str(x), x, verbose=False)
        prs.append(policy_reward)
        print("Policy - Gamma = " + str(x))
        print("Games won:" + str(get_rewards(env, policy_policy)))

        env.reset()
        value_policy, value_reward = value.main(env, 4, "4x4 Gamma= " + str(x), x, verbose=False)
        vrs.append(value_reward)
        print("Value - Gamma = " + str(x))
        print("Games won:" + str(get_rewards(env, value_policy)))

    for x in range(len(gammas)):
        plt.plot(range(1, len(vrs[x]) + 1), vrs[x], label="Gamma = " + str(gammas[x]))
    plt.title("Frozen Lake Value Iteration Reward vs Gamma")
    plt.xlabel("Iterations")
    plt.ylabel("Games won (out of 100)")
    plt.tight_layout()
    plt.legend()
    plt.savefig('FL VALUE vs Gamma Rewards 4x4 Chart.png')
    plt.close()
    plt.figure()

    for x in range(len(gammas)):
        plt.plot(range(1, len(prs[x]) + 1), prs[x], label="Gamma = " + str(gammas[x]))
    plt.title("Frozen Lake Policy Iteration Reward vs Gamma")
    plt.xlabel("Iterations")
    plt.ylabel("Games won (out of 100)")
    plt.tight_layout()
    plt.legend()
    plt.savefig('FL POLICY vs Gamma Rewards 4x4 Chart.png')
    plt.close()
    plt.figure()


    sizes = [4,10,25]
    probs = [.9,.9,.99]

    policy_rewards= []
    value_rewards = []

    for x in range(len(sizes)):
        print("\n\n\n")
        if x != 0:
            #From https: // reinforcement - learning4.fun / 2019 / 06 / 24 / create - frozen - lake - random - maps /
            random_map = generate_random_map(size=sizes[x], p=probs[x])
            env = gym.make("FrozenLake-v0", desc=random_map)
            env.seed(613)
            env.render()

            env.reset()

        name = str(sizes[x]) +"x"+str(sizes[x])

        p, r = policy.main(env, sizes[x], name, verbose = False)
        policy_rewards.append(r)
        print("Policy - size = " + name)
        print("Games won:" + str(get_rewards(env, p)))


        env.reset()
        p, r = value.main(env, sizes[x], name, verbose=False)
        value_rewards.append(r)
        print("Value - size = " + name)
        print("Games won:" + str(get_rewards(env, p)))


    for x in range(len(sizes)):
        plt.plot(range(1, len(policy_rewards[x]) + 1), policy_rewards[x], label="Size = " + str(sizes[x]) +"x"+str(sizes[x]))
    plt.title("Frozen Lake Policy Reward vs Problem Size")
    plt.xlabel("Iterations")
    plt.ylabel("Games won (out of 100)")
    plt.tight_layout()
    plt.legend()
    plt.savefig('FL POLICY vs Size Rewards Chart.png')
    plt.close()
    plt.figure()

    for x in range(len(sizes)):
        plt.plot(range(1, len(value_rewards[x]) + 1), value_rewards[x], label="Size = " + str(sizes[x]) +"x"+str(sizes[x]))
    plt.title("Frozen Lake Value Reward vs Problem Size")
    plt.xlabel("Iterations")
    plt.ylabel("Games won (out of 100)")
    plt.tight_layout()
    plt.legend()
    plt.savefig('FL VALUE vs Size Rewards Chart.png')
    plt.close()
    plt.figure()















