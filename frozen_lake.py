"""
Original work by:
- Ankit Choudhary
(https://www.analyticsvidhya.com/blog/2018/09/reinforcement-learning-model-based-planning-dynamic-programming/);
- Genevieve Hayes (https://gist.github.com/gkhayes/3d154e0505e31d6367be22ed3da2e955#file-mountain_car-py), and;
- Adesh Gautam (https://medium.com/swlh/introduction-to-reinforcement-learning-coding-q-learning-part-3-9778366a41c0).

Modified by Matthieu Divet (mdivet3@gatech.edu)
"""

import numpy as np
import gym
import matplotlib.pyplot as plt
import time

# Import and initialize Mountain Car Environment
env = gym.make('FrozenLake-v0')
env.reset()


def one_step_lookahead(environment, state, V, discount_factor):
    action_values = np.zeros(environment.nA)
    for action in range(environment.nA):
        for probability, next_state, reward, terminated in environment.P[state][action]:
            action_values[action] += 1 / len(environment.P[state][action]) * reward +\
                                     probability * discount_factor * V[next_state]
            # max(action_values) = max(sum [(1/nb_of_possible_s') * r(s) + T(s, a, s') * gamma * U(s')])
            #                    = r(s) + gamma * max(sum [T(s, a, s') * U(s')])
            # with max over the possible actions at state s and the sum over the possible s'
            # which is Bellman's equation, seen in class. Therefore we can use max(action_values) in value_iteration()
            # and argmax(action_values) in policy_iteration().
    return action_values


def policy_evaluation(policy, environment, discount_factor=1.0, theta=1e-9, max_iterations=1e9):
    # Number of evaluation iterations
    evaluation_iterations = 1
    # Initialize a value function for each state as zero
    V = np.zeros(environment.nS)
    # Repeat until change in value is below the threshold
    for i in range(int(max_iterations)):
        # Initialize a change of value function as zero
        delta = 0
        # Iterate though each state
        for state in range(environment.nS):
            # Initial a new value of current state
            v = 0
            # Try all possible actions which can be taken from this state
            for action, action_probability in enumerate(policy[state]):
                # Check how good next state will be
                for state_probability, next_state, reward, terminated in environment.P[state][action]:
                    # Calculate the expected value (action_probability will actually just be 0 or 1, except in the
                    # very first iteration of policy_iteration() - therefore we will indeed be evaluating the utility of
                    # the state as if the action to take was indeed the one given by the policy we're evaluating. This
                    # is linked to the absence of a max over the possible actions in the formula for the U_t(s))
                    v += action_probability * (1 / len(environment.P[state][action]) * reward +
                                               state_probability * discount_factor * V[next_state])
            # Calculate the absolute change of value function
            delta = max(delta, np.abs(V[state] - v))
            # Update value function
            V[state] = v
        evaluation_iterations += 1
        # Terminate if value change is insignificant
        if delta < theta:
            print('Policy evaluated in {} iterations.'.format(evaluation_iterations))
            return V


def policy_iteration(environment, discount_factor=1.0, max_iterations=1e9):
    # Start with a random policy
    # num states x num actions / num actions
    policy = np.ones([environment.nS, environment.nA]) / environment.nA
    # Initialize counter of evaluated policies
    evaluated_policies = 0
    # Repeat until convergence or critical number of iterations reached
    t0 = time.time()
    for i in range(int(max_iterations)):
        stable_policy = True
        # Evaluate current policy
        V = policy_evaluation(policy, environment, discount_factor=discount_factor)
        # Go through each state and try to improve actions that were taken (policy Improvement)
        for state in range(environment.nS):
            # Choose the best action in a current state under current policy
            current_action = np.argmax(policy[state])
            # Look one step ahead and evaluate if current action is optimal
            # We will try every possible action in a current state
            action_value = one_step_lookahead(environment, state, V, discount_factor)
            # Select a better action
            best_action = np.argmax(action_value)
            # If action changed
            if current_action != best_action:
                stable_policy = False
                # Greedy policy update
                # set all actions to 0 and the best action to 1 for current state in the matrix that represents
                # the policy
                policy[state] = np.eye(environment.nA)[best_action]
        evaluated_policies += 1
        # If the algorithm converged and policy is not changing anymore, then return final policy and value function
        if stable_policy:
            print('Evaluated {} policies.'.format(evaluated_policies))
            print('Evaluated in {} seconds.'.format(round(time.time()-t0, 2)))
            return policy, V


def value_iteration(environment, discount_factor=1.0, theta=1e-9, max_iterations=1e9):
    # Initialize state-value function with zeros for each environment state
    V = np.zeros(environment.nS)
    t0 = time.time()
    for i in range(int(max_iterations)):
        # Early stopping condition (if delta < theta then the algorithm will stop - before i = max_iterations)
        delta = 0
        # Update each state
        for state in range(environment.nS):
            # Do a one-step lookahead to calculate state-action values
            action_value = one_step_lookahead(environment, state, V, discount_factor)
            # Select best action to perform based on the highest state-action value
            best_action_value = np.max(action_value)
            # Calculate change in value (delta represents the maximum change in the utility estimation V at the current
            # iteration)
            delta = max(delta, np.abs(V[state] - best_action_value))
            # Update the value function for current state
            V[state] = best_action_value
            # Check if we can stop (theta is the convergence criterion, once delta is smaller than theta we consider
            # the utility function has converged - another iteration wouldn't improve its estimation by more than theta)
        if delta < theta:
            print('Value-iteration converged at iteration #{}.'.format(i))
            print('Converged after {} seconds.'.format(round(time.time() - t0, 2)))
            break

    # Create a deterministic policy using the optimal value function
    policy = np.zeros([environment.nS, environment.nA])
    for state in range(environment.nS):
        # One step lookahead to find the best action for this state
        action_value = one_step_lookahead(environment, state, V, discount_factor)
        # Select best action based on the highest state-action value
        best_action = np.argmax(action_value)
        # Update the policy to perform a better action at a current state (mark the best action to take at each state
        # with a 1 in the matrix that represents the policy)
        policy[state, best_action] = 1.0
    return policy, V


def play_episodes(environment, n_episodes, policy):
    wins = 0
    total_reward = 0
    for episode in range(n_episodes):
        terminated = False
        state = environment.reset()
        while not terminated:
            # Select best action to perform in a current state
            action = np.argmax(policy[state])
            # Perform an action an observe how environment acted in response
            next_state, reward, terminated, info = environment.step(action)
            # Summarize total reward
            total_reward += reward
            # Update current state
            state = next_state
            # Calculate number of wins over episodes
            if terminated and reward == 1.0:
                wins += 1
    average_reward = total_reward / n_episodes
    return wins, total_reward, average_reward


# Number of episodes to play
n_episodes = 10000
# Functions to find best policy
solvers = [('Policy Iteration', policy_iteration),
           ('Value Iteration', value_iteration)]
for iteration_name, iteration_func in solvers:
    # Load a Frozen Lake environment
    environment = gym.make('FrozenLake-v0')
    # Search for an optimal policy using policy iteration
    policy, V = iteration_func(environment.env)
    # Apply best policy to the real environment
    wins, total_reward, average_reward = play_episodes(environment, n_episodes, policy)
    print('{} :: policy found = {}'.format(iteration_name, policy))
    print('{} :: number of wins over {} episodes = {}'.format(iteration_name, n_episodes, wins))
    print('{} :: average reward over {} episodes = {} \n\n'.format(iteration_name, n_episodes, average_reward))


# Define Q-learning function
def QLearning(env, learning, discount, epsilon, min_eps, episodes, exploration="linear-decay"):
    # Determine size of discretized state space
    num_states = 16

    # Initialize Q table
    Q = np.random.uniform(low=-1, high=1, size=(num_states, env.action_space.n))

    # Initialize variables to track rewards
    reward_list = []
    ave_reward_list = []

    # Calculate episodic reduction in epsilon in case exploration == "linear-decay"
    reduction = (epsilon - min_eps) / episodes

    # If exploration == "greedy" then epsilon needs to be 0
    if exploration == "greedy":
        epsilon = 0

    # Run Q learning algorithm
    for i in range(episodes):
        # Initialize parameters
        done = False
        tot_reward, reward = 0, 0
        state = env.reset()

        while done is not True:
            # Render environment for last five episodes
            if i >= (episodes - 5):
                env.render()

            # Determine next action - epsilon greedy strategy (if epsilon == 0 this is just a greedy exploration)
            if np.random.random() < 1 - epsilon:
                action = np.argmax(Q[state, :])
            else:
                action = env.action_space.sample()

            # Get next state and reward
            state2, reward, done, info = env.step(action)

            # Allow for terminal states and render successful state
            if done:
                Q[state, action] = reward
                env.render()

            # Adjust Q value for current state
            else:
                delta = learning * (reward + discount * np.max(Q[state2, :]) - Q[state, action])
                Q[state, action] += delta

            # Update variables
            tot_reward += reward
            state = state2

        # Decay epsilon
        if epsilon > min_eps:
            if exploration == "linear-decay":
                epsilon -= reduction
            elif exploration == "exp-decay":
                epsilon *= 1 / 2
            # elif exploration == "greedy" or exploration == "epsilon-greedy" epsilon mudt not change

        # Track rewards
        reward_list.append(tot_reward)

        if (i + 1) % 100 == 0:
            ave_reward = np.mean(reward_list)
            ave_reward_list.append(ave_reward)
            reward_list = []

        if (i + 1) % 100 == 0:
            print('Episode {} Average Reward: {}'.format(i + 1, ave_reward))

    env.close()

    return ave_reward_list


# Run Q-learning algorithm
rewards = QLearning(env, 0.2, 0.9, 0.8, 0, 1000000, exploration="linear-decay")

# Plot Rewards
plt.plot(100 * (np.arange(len(rewards)) + 1), rewards)
plt.xlabel('Episodes')
plt.ylabel('Average Reward')
plt.title('Average Reward vs Episodes')
plt.savefig('Rewards_graphs/rewards_frozen_lake.jpg')
plt.close()
