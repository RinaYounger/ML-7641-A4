import numpy as np
import gym
import random
import time
import matplotlib.pyplot as plt
import gym
from gym import utils as utils
from gym.envs.toy_text.frozen_lake import generate_random_map
#From https://learning.oreilly.com/library/view/reinforcement-learning-algorithms/9781789131116/7c6dfed0-1180-49fe-84a0-ea62131b5947.xhtml



def eval_state_action(V, s, a, env, gamma):
    return np.sum([p * (rew + gamma * V[next_s]) for p, next_s, rew, _ in env.P[s][a]])


def policy_evaluation(V, policy, nS, env, gamma, eps=0.0001):
    '''
    Policy evaluation. Update the value function until it reach a steady state
    '''
    while True:
        delta = 0
        # loop over all states
        for s in range(nS):
            old_v = V[s]
            # update V[s] using the Bellman equation
            V[s] = eval_state_action(V, s, policy[s], env, gamma)
            delta = max(delta, np.abs(old_v - V[s]))

        if delta < eps:
            break


def policy_improvement(V, policy, nS, nA, env, gamma):
    '''
    Policy improvement. Update the policy based on the value function
    '''
    policy_stable = True
    counter = 0
    for s in range(nS):
        old_a = policy[s]
        # update the policy with the action that bring to the highest state value
        policy[s] = np.argmax([eval_state_action(V, s, a, env, gamma) for a in range(nA)])
        if old_a != policy[s]:
            policy_stable = False
            counter +=1

    return policy_stable, counter


def run_episodes(env, policy, num_games=100):
    '''
    Run some games to test a policy
    '''
    tot_rew = 0
    state = env.reset()

    for _ in range(num_games):
        done = False
        while not done:
            # select the action accordingly to the policy
            next_state, reward, done, _ = env.step(policy[state])

            state = next_state
            tot_rew += reward
            if done:
                state = env.reset()

    # print('Won %i of %i games!' % (tot_rew, num_games))
    return tot_rew

def main(env, n, name, gamma=.99, verbose = False):

    print("Policy " + name)

    random.seed(26)
    np.random.seed(26)
    utils.seeding.np_random(26)
    utils.seeding.hash_seed(26)
    utils.seeding.create_seed(26)

    nA = env.action_space.n
    nS = env.observation_space.n

    # initializing value function and policy
    V = np.zeros(nS)
    policy = np.zeros(nS)

    # some useful variable
    policy_stable = False
    it = 0
    rewards = []
    times = []
    deltas = []
    maxv = []
    meanv= []

    while not policy_stable:
        start = time.time()
        policy_evaluation(V, policy, nS, env, gamma)
        policy_stable, delta = policy_improvement(V, policy, nS, nA, env, gamma)
        end = time.time()


        it += 1

        total_reward = run_episodes(env, policy, 100)
        rewards.append(total_reward)
        times.append(end - start)
        deltas.append(delta)
        maxv.append(np.max(V))
        meanv.append(np.mean(V))

    times = np.cumsum(times)


    if verbose:
        # policy = policy.reshape((n, n))
        print('Converged after %i policy iterations' % (it))
        # run_episodes(env, policy)
        # print(V.reshape((n, n)))
        print(policy)


        plt.plot(range(1,len(rewards)+1), rewards, label = "POLICY Iter")
        plt.title("Frozen Lake POLICY iter vs Games Won")
        plt.xlabel("Iterations")
        plt.ylabel("Games won (out of 100)")
        plt.tight_layout()
        plt.legend()
        plt.savefig(name+' FL POLICY Rewards.png')
        plt.close()
        plt.figure()


        plt.plot(range(1,len(times)+1), times, label = "POLICY Iter")
        plt.title("Frozen Lake POLICY iter vs Time")
        plt.xlabel("Iterations")
        plt.ylabel("Time")
        plt.tight_layout()
        plt.legend()
        plt.savefig(name+' FL POLICY Time.png')
        plt.close()
        plt.figure()

        plt.plot(range(1, len(deltas) + 1), deltas, label="POLICY Iter")
        plt.title("Frozen Lake POLICY iter vs Delta Convergence")
        plt.xlabel("Iterations")
        plt.ylabel("Delta in the policy")
        plt.tight_layout()
        plt.legend()
        plt.savefig(name + 'FL POLICY Delta Convergence.png')
        plt.close()
        plt.figure()

        plt.plot(range(1, len(meanv) + 1), meanv, label="POLICY Iter")
        plt.title("Frozen Lake POLICY iter vs Mean Value")
        plt.xlabel("Iterations")
        plt.ylabel("Mean Value")
        plt.tight_layout()
        plt.legend()
        plt.savefig(name + 'FL POLICY Mean Value.png')
        plt.close()
        plt.figure()

        plt.plot(range(1, len(maxv) + 1), maxv, label="POLICY Iter")
        plt.title("Frozen Lake POLICY iter vs Max Value")
        plt.xlabel("Iterations")
        plt.ylabel("Max Value")
        plt.tight_layout()
        plt.legend()
        plt.savefig(name + 'FL POLICY Max Value.png')
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

