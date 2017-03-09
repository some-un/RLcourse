# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0

        # Write value iteration code here
        "*** YOUR CODE HERE ***"



    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        #
        # Retrieve all states (with the probabilities) that we can end up in after taking tha action
        state_prob_pairs = self.mdp.getTransitionStatesAndProbs(state, action)
        #

        sum_states_prob = 0
        sum_rewards = 0
        for state_prob_pair in state_prob_pairs:
            s = state_prob_pair[0]
            p = state_prob_pair[1]

            if self.getValue(s) is not None:

                value_s = self.getValue(s)

                sum_states_prob += (p * value_s)
                immediate_reward = self.mdp.getReward(state, action, s)
                sum_rewards += p * immediate_reward

        discounted_sum = sum_states_prob * self.discount

        final_value = sum_rewards + discounted_sum

        return final_value


        # temp = 0
        # for s,p in states_probs:
        #     temp += p*self.values[state]
        # #
        # # We multiply for the discount
        # temp = temp*self.discount
        # reward = self.mdp.getReward(state, action, states_probs[0][0])
        #
        # # This is the first version of the solution
        # # The reward for the first state we might end up in after taking the given action is considered
        # return temp + reward
        # #
        # # util.raiseNotDefined()

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        #
        #  initial implementation
        #
        if (self.mdp.isTerminal(state)):
            return None
        possibleActionsList = self.mdp.getPossibleActions(state)
        for a in possibleActionsList:
            self.values[a] = self.computeQValueFromValues(state,a)

        self.values.sortedKeys()
        bestaction = self.values.argMax()
        return self.values.argMax()


    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
