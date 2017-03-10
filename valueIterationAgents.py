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


        for i in range(self.iterations):
            previous_values = self.values.copy()
            for state in self.mdp.getStates():
                # Why None works and 0 doesn't ?!??!?!? After we compare a number (qvalue) with None...hmm
                maxvalue_per_action = None
                for action in self.mdp.getPossibleActions(state):
                    qvalue = 0
                    state_prob_tuple = self.mdp.getTransitionStatesAndProbs(state, action)
                    for next_state, prob in state_prob_tuple:
                        qvalue += prob * (
                            self.mdp.getReward(state, prob, next_state) + (self.discount * previous_values[next_state]))
                    if qvalue > maxvalue_per_action:
                        maxvalue_per_action = qvalue
                self.values[state] = (maxvalue_per_action or 0)




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
        max_value = None
        best_action = None
        for a in possibleActionsList:
            current_value = self.computeQValueFromValues(state,a)
            if current_value > max_value:
                max_value, best_action = current_value, a
        return best_action


    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
