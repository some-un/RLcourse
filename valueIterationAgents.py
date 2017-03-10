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
        #
        # Thanks to dictionary's default 0, we've value function initialization done right there
        #
#         for _ in xrange(iterations):
#             # we wouldn't use the iterator variable anyway and we can't do, e.g., # its=iterations # while its-- > 0 # :(
#             lastItVals = self.values.copy()
#             #
#             for s in self.mdp.getStates():
#                 #
#                 qvaluesForCurrentStateActions = util.Counter()
#                 #
#                 for a in self.mdp.getPossibleActions(s):
#                     tempqv = self.computeQValueFrom_previousIterationValues(s, a, lastItVals)
#                     if tempqv == None:
#                         qvaluesForCurrentStateActions[a] = 0
#                     else:
#                         qvaluesForCurrentStateActions[a] = tempqv 
#                 #
#                 print "unsorted: ", qvaluesForCurrentStateActions
#                 qvaluesForCurrentStateActions.sortedKeys()
#                 print "qvaluesForCurrentStateActions argMax: ", qvaluesForCurrentStateActions.argMax()
#                 if qvaluesForCurrentStateActions.argMax() == None:
#                     self.values[s] = 0
#                 else:
#                     
#                 self.values[s] = qvaluesForCurrentStateActions.argMax()
            
 
        for uselessIterator in range(self.iterations):
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
        #
        weightedVfvsSum = 0
        reward = 0
        # to get possible next state(s)
        for nextState, prob in self.mdp.getTransitionStatesAndProbs(state, action):
            reward += self.mdp.getReward(state, action, nextState) * prob
            #print ":computeQValueFromValues: nextState is: ", nextState, " | self.values[nextState] is: ", self.values[nextState]
            weightedVfvsSum += prob * self.getValue(nextState)
        #
        return ( reward + ( self.discount * weightedVfvsSum) ) # making the actual qvalue
    
    def computeQValueFrom_previousIterationValues(self, state, action, previousIterationValues):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        #
        weightedVfvsSum = 0
        reward = 0
        # to get possible next state(s)
        for nextState, prob in self.mdp.getTransitionStatesAndProbs(state, action):
            reward += self.mdp.getReward(state, action, nextState) * prob
            print ":computeQValueFrom_previousIterationValues: nextState is: ", nextState, " | self.values[nextState] is: ", previousIterationValues[nextState]
            weightedVfvsSum += prob * previousIterationValues[nextState]
        #
        return ( reward + ( self.discount * weightedVfvsSum) ) # making the actual qvalue


            # above crashes the system, is then this hack enough? ;) Seems to be, at least for this game
            #if self.values[nextState] == None or self.values[nextState] == 'exit': self.values[nextState] = 0 # nah, can be also west, ..., any string etc.
            # let's go for the following then:
            #if not isinstance(self.values[nextState], float): self.values[nextState] = 0
            
            #if not isinstance(self.values[nextState], float):
            #if (self.mdp.isTerminal(nextState)):
                #self.values[nextState] = 0
                #continue #as the "next", exit state value should be 0, resulting in the same for this weightedVfvsSum increment

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        #
        if (self.mdp.isTerminal(state)): return None
        #
        ExpectedValuesOfActions = util.Counter()
        #
        for a in self.mdp.getPossibleActions(state):
            ExpectedValuesOfActions[a] = self.computeQValueFromValues(state, a)
        #
        ExpectedValuesOfActions.sortedKeys()
        return ExpectedValuesOfActions.argMax()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
