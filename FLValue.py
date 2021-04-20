import numpy as np
import gym
import random
import time
import matplotlib.pyplot as plt
from gym import utils as utils
from gym.envs.toy_text.frozen_lake import generate_random_map

#from https://learning.oreilly.com/library/view/reinforcement-learning-algorithms/9781789131116/7c6dfed0-1180-49fe-84a0-ea62131b5947.xhtml

def eval_state_action(V, s, a, env, gamma):
    return np.sum([p * (rew + gamma * V[next_s]) for p, next_s, rew, _ in env.P[s][a]])


def value_iteration(eps,env, nS, nA, gamma):
    '''
    Value iteration algorithm
    '''
    V = np.zeros(nS)
    it = 0

    times = []
    rewards = []
    deltas = []
    maxv = []
    meanv = []

    while True:
        delta = 0

        start = time.time()
        # update the value of each state using as "policy" the max operator
        for s in range(nS):
            old_v = V[s]
            V[s] = np.max([eval_state_action(V, s, a, env, gamma) for a in range(nA)])
            delta = max(delta, np.abs(old_v - V[s]))

        end = time.time()


        if delta < eps:
            # print(policy)
            break
        # else:
            # print('Iter:', it, ' delta:', np.round(delta, 5))
        it += 1

        total_reward = run_episodes(env, V, nA, 100, gamma)

        rewards.append(total_reward)
        times.append(end-start)
        deltas.append(delta)
        meanv.append(np.mean(V))
        maxv.append(np.max(V))

    times = np.cumsum(times)
    print("Converged in iterations: " + str(it))

    return V, rewards, times, deltas, maxv, meanv


def extract_policy(value_table, env, gamma):
    policy = np.zeros(env.observation_space.n)
    for state in range(env.observation_space.n):
        Q_table = np.zeros(env.action_space.n)
        for action in range(env.action_space.n):
            for next_sr in env.P[state][action]:
                trans_prob, next_state, reward_prob, _ = next_sr
                Q_table[action] += (trans_prob * (reward_prob + gamma * value_table[next_state]))
        policy[state] = np.argmax(Q_table)

    return policy


def run_episodes(env, V, nA, num_games, gamma):
    tot_rew = 0
    state = env.reset()


    for _ in range(num_games):
        done = False

        while not done:
            # choose the best action using the value function
            action = np.argmax([eval_state_action(V, state, a, env, gamma) for a in range(nA)]) #(11)
            next_state, reward, done, _ = env.step(action)
            state = next_state
            tot_rew += reward
            if done:
                state = env.reset()

    # print('Won %i of %i games!'%(tot_rew, num_games))
    return tot_rew

def main(env, n, name, gamma=.99, verbose = False):
    print("Value " + name)

    random.seed(26)
    np.random.seed(26)
    utils.seeding.np_random(26)
    utils.seeding.hash_seed(26)
    utils.seeding.create_seed(26)

    nA = env.action_space.n
    nS = env.observation_space.n


    V, rewards, times, deltas, maxv, meanv = value_iteration(0.0001, env, nS, nA, gamma)



    optimal_policy = extract_policy(V, env,gamma)
    print(optimal_policy)


    if verbose:
        print(optimal_policy.reshape((n, n)))

        plt.plot(range(1, len(rewards)+1), rewards, label="Value Iter")
        plt.title("Frozen Lake VALUE iter vs Games Won")
        plt.xlabel("Iterations")
        plt.ylabel("Games won (out of 100)")
        plt.tight_layout()
        plt.legend()
        plt.savefig(name+'FL VALUE Reward.png')
        plt.close()
        plt.figure()

        plt.plot(range(1, len(times)+1), times, label="Value Iter")
        plt.title("Frozen Lake VALUE iter vs Time")
        plt.xlabel("Iterations")
        plt.ylabel("Time")
        plt.tight_layout()
        plt.legend()
        plt.savefig(name+'FL VALUE Time.png')
        plt.close()
        plt.figure()

        plt.plot(range(1, len(deltas) + 1), deltas, label="Value Iter")
        plt.title("Frozen Lake VALUE iter vs Delta Convergence")
        plt.xlabel("Iterations")
        plt.ylabel("delta")
        plt.tight_layout()
        plt.legend()
        plt.savefig(name + 'FL VALUE Delta Convergence.png')
        plt.close()
        plt.figure()

        plt.plot(range(1, len(meanv) + 1), meanv, label="Value Iter")
        plt.title("Frozen Lake VALUE iter vs Mean Value")
        plt.xlabel("Iterations")
        plt.ylabel("Mean Value")
        plt.tight_layout()
        plt.legend()
        plt.savefig(name + 'FL VALUE Mean Value.png')
        plt.close()
        plt.figure()

        plt.plot(range(1, len(maxv) + 1), maxv, label="Value Iter")
        plt.title("Frozen Lake VALUE iter vs Max Value")
        plt.xlabel("Iterations")
        plt.ylabel("Max Value")
        plt.tight_layout()
        plt.legend()
        plt.savefig(name + 'FL VALUE Max Value.png')
        plt.close()
        plt.figure()

        print("rewards " + str(rewards[-1]))
        print("mean v " + str(meanv[-1]))
        print("max v " + str(maxv[-1]))
        print("time " + str(times[-1]))

    return optimal_policy, rewards



if __name__ == '__main__':
    random.seed(26)
    utils.seeding.np_random(26)
    utils.seeding.hash_seed(26)
    utils.seeding.create_seed(26)
    np.random.seed(26)


