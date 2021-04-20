import numpy as np
import gym
import time
import matplotlib.pyplot as plt
import random
import copy
from gym import utils as utils


#From https://learning.oreilly.com/library/view/reinforcement-learning-algorithms/9781789131116/5b0a7336-a2f4-489e-a47e-7d7601a592f2.xhtml


def eps_greedy(Q, s, eps=0.1):
    '''
    Epsilon greedy policy
    '''
    if np.random.uniform(0, 1) < eps:
        # Choose a random action
        return np.random.randint(Q.shape[1])
    else:
        # Choose the action of a greedy policy
        return greedy(Q, s)


def greedy(Q, s):
    '''
    Greedy policy
    return the index corresponding to the maximum action-state value
    '''
    return np.argmax(Q[s])


def run_episodes(env, Q, num_episodes=100, to_print=False):
    '''
    Run some episodes to test the policy
    '''
    tot_rew = 0
    state = env.reset()

    for _ in range(num_episodes):
        done = False
        game_rew = 0

        while not done:
            # select a greedy action
            next_state, rew, done, _ = env.step(greedy(Q, state))

            state = next_state
            game_rew += rew
            if done:
                state = env.reset()
                tot_rew += game_rew

    if to_print:
        print('Won %i of %i games!' % (tot_rew, num_episodes))

    return tot_rew






def Q_learning(env, lr=0.01, num_episodes=10000, eps=0.3, gamma=0.99, eps_decay=0.00005):
    nA = env.action_space.n
    nS = env.observation_space.n

    # Initialize the Q matrix
    # Q: matrix nS*nA where each row represent a state and each colums represent a different action
    Q = np.zeros((nS, nA))
    games_reward = []
    test_rewards = []
    times = []
    maxv = []
    meanv = []
    deltas = []

    for ep in range(num_episodes):
        state = env.reset()
        done = False
        tot_rew = 0
        oldQ = np.copy(Q)

        # decay the epsilon value until it reaches the threshold of 0.01
        if eps > 0.01:
            eps -= eps_decay

        # loop the main body until the environment stops
        start = time.time()
        while not done:
            # select an action following the eps-greedy policy
            action = eps_greedy(Q, state, eps)

            next_state, rew, done, _ = env.step(action)  # Take one step in the environment

            # Q-learning update the state-action value (get the max Q value for the next state)
            Q[state][action] = Q[state][action] + lr * (rew + gamma * np.max(Q[next_state]) - Q[state][action])


            state = next_state
            tot_rew += rew
            if done:
                games_reward.append(tot_rew)
        end = time.time()
        print(ep)


        # Test the policy every 300 episodes and print the results
        # if (ep % 300) == 0:
        test_rew = run_episodes(env, Q, 100)
        # print("Episode:{:5d}  Eps:{:2.4f}  Num Games Won:{:5d}".format(ep, eps, test_rew))
        test_rewards.append(test_rew)
        times.append((end-start))
        maxv.append(np.max(Q))
        meanv.append(np.mean(Q))
        deltas.append(np.sum(np.abs(oldQ - Q)))
        if(end-start) > 1:
            break


    times = np.cumsum(times)
    return Q, test_rewards,times, maxv, deltas, meanv



def extract_policy(Q_table, env):
    policy = np.zeros(env.observation_space.n)
    for state in range(env.observation_space.n):
        policy[state] = np.argmax(Q_table[state])

    return policy



def main(env, n, name):

    print("Q Learning " + name)

    random.seed(26)
    np.random.seed(26)
    utils.seeding.np_random(26)
    utils.seeding.hash_seed(26)
    utils.seeding.create_seed(26)

    # Q_qlearning, rewards, times = Q_learning(env, lr=.1, num_episodes=5000, eps=.5, gamma=0.99, eps_decay=0.000001)
    Q_qlearning, rewards, times, maxv, deltas, meanv = Q_learning(env, lr=.1, num_episodes=3000, eps=.5, gamma=0.99, eps_decay=0.0001)
    policy = extract_policy(Q_qlearning, env)#.reshape((n, n))
    print(policy)

    plt.plot(range(1, len(rewards)+1), rewards, label="Q")
    plt.title("Frozen Lake Q iter vs Games Won")
    plt.xlabel("Iterations")
    plt.ylabel("Games won (out of 100)")
    plt.tight_layout()
    plt.legend()
    plt.savefig(name+' FL Q Reward.png')
    plt.close()
    plt.figure()

    plt.plot(range(1, len(times)+1), times, label="Q")
    plt.title("Frozen Lake Q iter vs Time")
    plt.xlabel("Iterations")
    plt.ylabel("Time")
    plt.tight_layout()
    plt.legend()
    plt.savefig(name+' FL Q Time.png')
    plt.close()
    plt.figure()

    plt.plot(range(1, len(maxv) + 1), maxv, label="Q")
    plt.title("Frozen Lake Q iter vs Max Q Value")
    plt.xlabel("Iterations")
    plt.ylabel("Max Utility")
    plt.tight_layout()
    plt.legend()
    plt.savefig(name + ' FL Q Max Utility.png')
    plt.close()
    plt.figure()

    plt.plot(range(1, len(deltas) + 1), deltas, label="Q")
    plt.title("Frozen Lake Q iter vs Delta")
    plt.xlabel("Iterations")
    plt.ylabel("Delta")
    plt.tight_layout()
    plt.legend()
    plt.savefig(name + ' FL Q Delta.png')
    plt.close()
    plt.figure()

    plt.plot(range(1, len(meanv) + 1), meanv, label="Q")
    plt.title("Frozen Lake Q iter vs Mean Q Value")
    plt.xlabel("Iterations")
    plt.ylabel("Mean Utility")
    plt.tight_layout()
    plt.legend()
    plt.savefig(name + ' FL Q Mean Utility.png')
    plt.close()
    plt.figure()

    print("rewards " + str(rewards[-1]))
    print("mean v " + str(meanv[-1]))
    print("max v " + str(maxv[-1]))
    print("time " + str(times[-1]))


    return policy, rewards


if __name__ == '__main__':

    random.seed(26)
    utils.seeding.np_random(26)
    utils.seeding.hash_seed(26)
    utils.seeding.create_seed(26)
    np.random.seed(26)

    env = gym.make('FrozenLake-v0')
    env.seed(613)


    epsilons = [.05,.1,.5,1,1.5]
    colors = ['r','g','c','m','y']
    r = []
    t = []
    ma = []
    d = []
    me = []

    for x in epsilons:

        Q_qlearning, rewards, times, maxv, deltas, meanv = Q_learning(env, lr=.1, num_episodes=3000, eps=x, gamma=0.99, eps_decay=0.0001)
        r.append(rewards)
        t.append(times)
        ma.append(maxv)
        d.append(deltas)
        me.append(meanv)


    for x in range(len(r)):
        plt.plot(range(0, len(r[x])), r[x], label="Epsilon = " + str(epsilons[x]), color = colors[x])
    plt.title("Frozen Lake Epsilon vs Reward")
    plt.xlabel("Iterations")
    plt.ylabel("Games Won (Out of 100)")
    plt.tight_layout()
    plt.legend()
    plt.savefig('FL Q Epsilon Rewards.png')
    plt.close()
    plt.figure()

    for x in range(len(r)):
        plt.plot(range(0, len(t[x])), t[x], label="Epsilon = " + str(epsilons[x]), color = colors[x])
    plt.title("Frozen Lake Epsilon vs Time")
    plt.xlabel("Iterations")
    plt.ylabel("Time")
    plt.tight_layout()
    plt.legend()
    plt.savefig('FL Q Epsilon Times.png')
    plt.close()
    plt.figure()

    for x in range(len(ma)):
        plt.plot(range(0, len(ma[x])), ma[x], label="Epsilon = " + str(epsilons[x]), color=colors[x])
    plt.title("Frozen Lake Epsilon vs Maximum Q")
    plt.xlabel("Iterations")
    plt.ylabel("Q")
    plt.tight_layout()
    plt.legend()
    plt.savefig('FL Q Epsilon Max Q.png')
    plt.close()
    plt.figure()

    for x in range(len(me)):
        plt.plot(range(0, len(me[x])), me[x], label="Epsilon = " + str(epsilons[x]), color=colors[x])
    plt.title("Frozen Lake Epsilon vs Mean Q")
    plt.xlabel("Iterations")
    plt.ylabel("Q")
    plt.tight_layout()
    plt.legend()
    plt.savefig('FL Q Epsilon Mean Q.png')
    plt.close()
    plt.figure()

    for x in range(len(d)):
        plt.plot(range(0, len(d[x])), d[x], label="Epsilon = " + str(epsilons[x]), color=colors[x])
    plt.title("Frozen Lake Epsilon vs Delta")
    plt.xlabel("Iterations")
    plt.ylabel("Delta")
    plt.tight_layout()
    plt.legend()
    plt.savefig('FL Q Epsilon Delta.png')
    plt.close()
    plt.figure()


    epsilons = [.01, .001, .0001, .00001, .000001]
    colors = ['r', 'g', 'c', 'm', 'y']
    r = []
    t = []
    ma = []
    d = []
    me = []

    for x in epsilons:
        Q_qlearning, rewards, times, maxv, deltas, meanv = Q_learning(env, lr=.1, num_episodes=3000, eps=.5, gamma=0.99, eps_decay=x)
        r.append(rewards)
        t.append(times)
        ma.append(maxv)
        d.append(deltas)
        me.append(meanv)

    for x in range(len(r)):
        plt.plot(range(0, len(r[x])), r[x], label="Epsilon Decay = " + str(epsilons[x]), color=colors[x])
    plt.title("Frozen Lake (e = .5) Epsilon Decay vs Reward")
    plt.xlabel("Iterations")
    plt.ylabel("Games Won (Out of 100)")
    plt.tight_layout()
    plt.legend()
    plt.savefig('FL Q Epsilon Decay Rewards.png')
    plt.close()
    plt.figure()

    for x in range(len(r)):
        plt.plot(range(0, len(t[x])), t[x], label="Epsilon Decay= " + str(epsilons[x]), color=colors[x])
    plt.title("Frozen Lake (e=.5) Epsilon Decay vs Time")
    plt.xlabel("Iterations")
    plt.ylabel("Time")
    plt.tight_layout()
    plt.legend()
    plt.savefig('FL Q Epsilon Decay Times.png')
    plt.close()
    plt.figure()

    for x in range(len(ma)):
        plt.plot(range(0, len(ma[x])), ma[x], label="Epsilon = " + str(epsilons[x]), color=colors[x])
    plt.title("Frozen Lake Epsilon Decay vs Maximum Q")
    plt.xlabel("Iterations")
    plt.ylabel("Q")
    plt.tight_layout()
    plt.legend()
    plt.savefig('FL Q Epsilon Decay Max Q.png')
    plt.close()
    plt.figure()

    for x in range(len(me)):
        plt.plot(range(0, len(me[x])), me[x], label="Epsilon = " + str(epsilons[x]), color=colors[x])
    plt.title("Frozen Lake Epsilon Decay vs Mean Q")
    plt.xlabel("Iterations")
    plt.ylabel("Q")
    plt.tight_layout()
    plt.legend()
    plt.savefig('FL Q Epsilon Decay Mean Q.png')
    plt.close()
    plt.figure()

    for x in range(len(d)):
        plt.plot(range(0, len(d[x])), d[x], label="Epsilon = " + str(epsilons[x]), color=colors[x])
    plt.title("Frozen Lake Epsilon Decay vs Delta")
    plt.xlabel("Iterations")
    plt.ylabel("Delta")
    plt.tight_layout()
    plt.legend()
    plt.savefig('FL Q Epsilon Decay Delta.png')
    plt.close()
    plt.figure()


    gammas = [.5, .8, .95, .99]
    colors = ['r', 'g', 'c', 'm', 'y']
    r = []
    t = []
    ma = []
    d = []
    me = []

    for x in gammas:
        Q_qlearning, rewards, times, maxv, deltas, meanv = Q_learning(env, lr=.1, num_episodes=3000, eps=.5, gamma=x, eps_decay=0.0001)
        r.append(rewards)
        t.append(times)
        ma.append(maxv)
        d.append(deltas)
        me.append(meanv)

    for x in range(len(r)):
        plt.plot(range(0, len(r[x])), r[x], label="Gamma = " + str(gammas[x]), color=colors[x])
    plt.title("Frozen Lake Gamma vs Reward")
    plt.xlabel("Iterations")
    plt.ylabel("Games Won (Out of 100)")
    plt.tight_layout()
    plt.legend()
    plt.savefig('FL Q Gamma Reward new.png')
    plt.close()
    plt.figure()

    for x in range(len(r)):
        plt.plot(range(0, len(t[x])), t[x], label="Gamma = " + str(gammas[x]), color=colors[x])
    plt.title("Frozen Lake Gamma vs Time")
    plt.xlabel("Iterations")
    plt.ylabel("Time")
    plt.tight_layout()
    plt.legend()
    plt.savefig('FL Q Gamma Time new.png')
    plt.close()
    plt.figure()

    for x in range(len(ma)):
        plt.plot(range(0, len(ma[x])), ma[x], label="Gamma =  " + str(gammas[x]), color=colors[x])
    plt.title("Frozen Lake Gamma vs Maximum Q")
    plt.xlabel("Iterations")
    plt.ylabel("Q")
    plt.tight_layout()
    plt.legend()
    plt.savefig('FL Q Gamma Max Q.png')
    plt.close()
    plt.figure()

    for x in range(len(me)):
        plt.plot(range(0, len(me[x])), me[x], label="Gamma =  " + str(gammas[x]), color=colors[x])
    plt.title("Frozen Lake Gamma vs Mean Q")
    plt.xlabel("Iterations")
    plt.ylabel("Q")
    plt.tight_layout()
    plt.legend()
    plt.savefig('FL Q Gamma Mean Q.png')
    plt.close()
    plt.figure()

    for x in range(len(d)):
        plt.plot(range(0, len(d[x])), d[x], label="Gamma =  " + str(gammas[x]), color=colors[x])
    plt.title("Frozen Lake Gamma vs Delta")
    plt.xlabel("Iterations")
    plt.ylabel("Delta")
    plt.tight_layout()
    plt.legend()
    plt.savefig('FL Q Gamma Delta.png')
    plt.close()
    plt.figure()

