"""
Original work by Ankit Choudhary
(https://www.analyticsvidhya.com/blog/2018/09/reinforcement-learning-model-based-planning-dynamic-programming/)
and by Genevieve Hayes (https://gist.github.com/gkhayes/3d154e0505e31d6367be22ed3da2e955#file-mountain_car-py)
Modified by Matthieu Divet (mdivet3@gatech.edu)
"""

import numpy as np
import gym
import matplotlib.pyplot as plt
import time

# Import and initialize Mountain Car Environment
environment = gym.make('MountainCar-v0')
environment.reset()
mountain_car_nA = 3
# discretization of the environment's state space into a 91 x 71 space = 6461 possible states
# (position in the range [-60, 30]/50 and velocity in the range [-35, 35]/500)
mountain_car_discretized_nS = np.prod(np.round((environment.observation_space.high - environment.observation_space.low)
                                               * np.array([50, 500]), 0).astype(int) + 1)


def one_step_lookahead(env, state, utility, discount_factor):
    """
    max(action_values) = max(sum [(1/nb_of_possible_s') * r(s) + T(s, a, s') * gamma * U(s')])
                       = r(s) + gamma * max(sum [T(s, a, s') * U(s')])
    with max over the possible actions at state s and the sum over the possible s'
    and where T(s, a, s') = 1 because of the following piece of code from the definition of env.step(action):
            position, velocity = self.state
            velocity += (action-1)*self.force + math.cos(3*position)*(-self.gravity)
            velocity = np.clip(velocity, -self.max_speed, self.max_speed)
            position += velocity
            position = np.clip(position, self.min_position, self.max_position)
            if (position==self.min_position and velocity<0): velocity = 0
    which means that s' (given by [position, velocity]) is entirely defined by s and a.
    This is Bellman's equation, seen in class. Therefore we can use max(action_values) in value_iteration()
    and argmax(action_values) in policy_iteration().


    Also, this environment does not have a transition matrix like Frozen Lake. So, to analyze state s, we convert its
    index into the corresponding values (position, velocity) and we then set the environment's state to be
    [position, velocity). Therefore, calling environment.step(action) will now give the next state, the reward,
    if the next state is final and info based on the state we set.
    """
    action_values = np.zeros(mountain_car_nA)
    position = float((state // 71) - 60) / 50
    velocity = float(state - 71 * (state // 71) - 35) / 500
    env.state = np.array([position, velocity])
    for action in range(mountain_car_nA):
        next_state, reward, done, info = env.step(action)
        next_state_index = ((np.round(next_state[0] * 50, 0).astype(int) + 60) * 71) \
                           + np.round(next_state[1] * 500, 0).astype(int) + 35
        action_values[action] += reward + discount_factor * utility[next_state_index]
    return action_values


def policy_evaluation(pol, env, discount_factor=1.0, theta=2, max_iterations=1e9):
    # Number of evaluation iterations
    evaluation_iterations = 1
    # Initialize a value function for each state as zero
    utility = np.zeros(mountain_car_discretized_nS)
    # Repeat until change in value is below the threshold
    for i in range(int(max_iterations)):
        # Initialize a change of value function as zero
        delta = 0
        # Iterate though each state
        for state in range(mountain_car_discretized_nS):
            position = float((state // 71) - 60) / 50
            velocity = float(state - 71 * (state // 71) - 35) / 500
            env.state = np.array([position, velocity])
            # Initial a new value of current state
            u = 0
            # Try all possible actions which can be taken from this state
            for action, action_probability in enumerate(pol[state]):
                # Check how good next state will be
                # Calculate the expected value (action_probability will actually just be 0 or 1, except in the
                # very first iteration of policy_iteration() - therefore we will indeed be evaluating the utility of
                # the state as if the action to take was indeed the one given by the policy we're evaluating. This
                # is linked to the absence of a max over the possible actions in the formula for the U_t(s))
                next_state, reward, done, info = env.step(action)
                next_state_index = ((np.round(next_state[0] * 50, 0).astype(int) + 60) * 71) \
                                   + np.round(next_state[1] * 500, 0).astype(int) + 35
                u += action_probability * (reward + discount_factor * utility[next_state_index])
            # Calculate the absolute change of value function
            delta = max(delta, np.abs(utility[state] - u))
            # Update value function
            utility[state] = u
        evaluation_iterations += 1
        # Terminate if value change is insignificant
        if delta < theta:
            print('Policy evaluated in {} iterations.'.format(evaluation_iterations))
            return utility


def policy_iteration(envi, discount_factor=1.0, max_iterations=1e9):
    # Start with a random policy
    # num states x num actions / num actions
    poli = np.ones([mountain_car_discretized_nS, mountain_car_nA]) / mountain_car_nA
    # Initialize counter of evaluated policies
    evaluated_policies = 0
    # Repeat until convergence or critical number of iterations reached
    t0 = time.time()
    for i in range(int(max_iterations)):
        stable_policy = True
        changes_in_policy = 0
        # Evaluate current policy
        U = policy_evaluation(poli, envi, discount_factor=discount_factor)
        # Go through each state and try to improve actions that were taken (policy Improvement)
        for state in range(mountain_car_discretized_nS):
            # Choose the best action in a current state under current policy
            current_action = np.argmax(poli[state])
            # Look one step ahead and evaluate if current action is optimal
            # We will try every possible action in a current state
            action_value = one_step_lookahead(envi, state, U, discount_factor)
            # Select a better action
            best_action = np.argmax(action_value)
            # If action changed
            if current_action != best_action:
                stable_policy = False
                changes_in_policy += 1
                # Greedy policy update
                # set all actions to 0 and the best action to 1 for current state in the matrix that represents
                # the policy
                poli[state] = np.eye(mountain_car_nA)[best_action]
        evaluated_policies += 1
        print('{} actions changed.'.format(changes_in_policy))
        # If the algorithm converged and policy is not changing anymore, then return final policy and value function
        if stable_policy:
            print('Evaluated {} policies.'.format(evaluated_policies))
            print('Evaluated in {} seconds.'.format(round(time.time() - t0, 2)))
            return poli, U


def value_iteration(env, discount_factor=1.0, theta=2, max_iterations=1e9):
    # Initialize state-value function with zeros for each environment state
    utility = np.zeros(mountain_car_discretized_nS)
    t0 = time.time()
    for i in range(int(max_iterations)):
        # Early stopping condition
        delta = 0
        # Update each state
        for state in range(mountain_car_discretized_nS):
            # Do a one-step lookahead to calculate state-action values
            action_value = one_step_lookahead(env, state, utility, discount_factor)
            # Select best action to perform based on the highest state-action value
            best_action_value = np.max(action_value)
            # Calculate change in value
            delta = max(delta, np.abs(utility[state] - best_action_value))
            # Update the value function for current state
            utility[state] = best_action_value
            # Check if we can stop
            # the utility function has converged - another iteration wouldn't improve its estimation by more than theta)
        if delta < theta:
            print('Value-iteration converged at iteration #{}.'.format(i))
            print('Converged after {} seconds.'.format(round(time.time() - t0, 2)))
            break

    # Create a deterministic policy using the optimal value function
    pol = np.zeros([mountain_car_discretized_nS, mountain_car_nA])
    for state in range(mountain_car_discretized_nS):
        # One step lookahead to find the best action for this state
        action_value = one_step_lookahead(env, state, utility, discount_factor)
        # Select best action based on the highest state-action value
        best_action = np.argmax(action_value)
        # Update the policy to perform a better action at a current state
        pol[state, best_action] = 1.0
    return pol, utility


def play_episodes(env, nb_episodes, pol):
    win = 0
    tot_reward = 0
    for episode in range(nb_episodes):
        terminated = False
        state = env.reset()
        while not terminated:
            # Select best action to perform in a current state
            action = np.argmax(pol[state])
            # Perform an action an observe how environment acted in response
            next_state, reward, terminated, info = env.step(action)
            # Summarize total reward
            tot_reward += reward
            # Update current state
            state = next_state
            # Calculate number of wins over episodes
            if terminated and reward == 1.0:
                win += 1
    avg_reward = tot_reward / nb_episodes
    return win, tot_reward, avg_reward


# Number of episodes to play
n_episodes = 10000
# Functions to find best policy
solvers = [('Policy Iteration', policy_iteration),
           ('Value Iteration', value_iteration)]
for iteration_name, iteration_func in solvers:
    # Load a Mountain Car environment
    environment = gym.make('MountainCar-v0')
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
    num_states = (env.observation_space.high - env.observation_space.low) * \
                 np.array([100, 1000])
    num_states = np.round(num_states, 0).astype(int) + 1

    # Initialize Q table
    Q = np.random.uniform(low=-1, high=1,
                          size=(num_states[0], num_states[1],
                                env.action_space.n))

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

        # Discretize state
        state_adj = (state - env.observation_space.low) * np.array([100, 1000])
        state_adj = np.round(state_adj, 0).astype(int)

        while done is not True:
            # Render environment for last five episodes
            if i >= (episodes - 5):
                env.render()

            # Determine next action - epsilon greedy strategy (if epsilon == 0 this is just a greedy exploration)
            if np.random.random() < 1 - epsilon:
                action = np.argmax(Q[state_adj[0], state_adj[1]])
            else:
                action = np.random.randint(0, env.action_space.n)

            # Get next state and reward
            state2, reward, done, info = env.step(action)

            # Discretize state2
            state2_adj = (state2 - env.observation_space.low) * np.array([100, 1000])
            state2_adj = np.round(state2_adj, 0).astype(int)

            # Allow for terminal states and render successful state
            if done and state2[0] >= 0.5:
                Q[state_adj[0], state_adj[1], action] = reward
                env.render()

            # Adjust Q value for current state
            else:
                delta = learning * (reward +
                                    discount * np.max(Q[state2_adj[0],
                                                        state2_adj[1]]) -
                                    Q[state_adj[0], state_adj[1], action])
                Q[state_adj[0], state_adj[1], action] += delta

            # Update variables
            tot_reward += reward
            state_adj = state2_adj

        # Decay epsilon
        if epsilon > min_eps:
            if exploration == "linear-decay":
                epsilon -= reduction
            elif exploration == "exp-decay":
                epsilon *= 1/2
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
rewards = QLearning(environment, 0.2, 0.9, 0.8, 0, 40000, exploration="linear-decay")

# Plot Rewards
plt.plot(100 * (np.arange(len(rewards)) + 1), rewards)
plt.xlabel('Episodes')
plt.ylabel('Average Reward')
plt.title('Average Reward vs Episodes')
plt.savefig('Rewards_graphs/rewards_mountain_car.jpg')
plt.close()
