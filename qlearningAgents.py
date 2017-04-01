# qlearningAgents.py
# ------------------
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


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random,util,math

class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)
        #
        self.qvalues = util.Counter() #qv[(s,a)] = 1
        self.transprobs = util.Counter()
        self.EligTrVal = util.Counter()
        self.lambdaVal = 0.5

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        return self.qvalues[(state,action)]


    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        legalActions = self.getLegalActions(state)
        if len(legalActions) == 0:
            return 0.0
        qvaluesForS = util.Counter()
        for a in legalActions:
            qvaluesForS[a] = self.getQValue(state,a)
        qvaluesForS.sortedKeys()
        return qvaluesForS[qvaluesForS.argMax()]

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        legalActions = self.getLegalActions(state)
        if len(legalActions) == 0:
            return None
        qvaluesForS = util.Counter()
        for a in legalActions:
            qvaluesForS[a] = self.getQValue(state,a)
        sortedQvalues = qvaluesForS.sortedKeys()
        bestQvalueFirstKey = sortedQvalues[0]
        bestQvalue = qvaluesForS[bestQvalueFirstKey]
        bestQvalueActionsList = []
        bestActionToTakeFromS = bestQvalueFirstKey
        bestQvalueActionsList.append(bestQvalueFirstKey)
        for a in legalActions:
            if qvaluesForS[a] == bestQvalue and a != bestQvalueFirstKey:
                bestQvalueActionsList.append(a)
        if len(bestQvalueActionsList) > 1:
            # we've a tie here, sir!
            bestActionToTakeFromS = random.choice(bestQvalueActionsList)
        return bestActionToTakeFromS

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        legalActions = self.getLegalActions(state)
        if len(legalActions) == 0:
            return None
        if util.flipCoin(self.epsilon):
            return random.choice(legalActions)
        # implicit else here, of course
        return self.computeActionFromQValues(state)

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        bestActionForTheNextState = self.computeActionFromQValues(nextState)
        #
        nextStateQval = None
        if bestActionForTheNextState is None:
            nextStateQval = 0
        else:
            nextStateQval = self.getQValue(nextState, bestActionForTheNextState)
        delta = reward + (self.discount * nextStateQval) - self.getQValue(state, action)
        #
        self.EligTrVal[(state,action)] += 1
        # if the next state is a terminal one
        if len(self.getLegalActions(nextState)) == 0:
            # reset the eligibility trace
            self.EligTrVal[(state,action)] = 0
        #
        for sPrime, aPrime in self.qvalues:
            self.qvalues[(sPrime, aPrime)] += self.alpha * delta * self.EligTrVal[(sPrime, aPrime)]
            self.EligTrVal[(sPrime,aPrime)] *= self.discount * self.lambdaVal
        #
        '''
        sPrime = nextState
        #
        allLegalActionsForSPrime = self.getLegalActions(sPrime)
        #
        for aPrime in allLegalActionsForSPrime:
            self.qvalues[(sPrime,aPrime)] += self.alpha * delta * self.EligTrVal[(sPrime,aPrime)]
            self.EligTrVal[(sPrime,aPrime)] *= self.discount * self.lambdaVal
        #
        '''
        #delta = reward + (self.discount * self.getQValue(nextState, bestActionForTheNextState)) - self.getQValue(state, action)
        #self.qvalues[(state,action)] = self.getQValue(state, action) + (self.alpha * delta)

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        featuresDict = self.featExtractor.getFeatures(state,action)
        #
        qApproxValue = 0
        for key_f in featuresDict:
            # if we keep weights per (state and action), we don't actually approximate reducing the representation of the problem, it's still too fine grained 
            #qApproxValue += self.weights[(state,action,key_f)] * featuresDict[key_f]
            qApproxValue += self.weights[key_f] * featuresDict[key_f]
        return qApproxValue

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        #
        '''
        # Approximate Q-learning code below
        bestActionForTheNextState = self.computeActionFromQValues(nextState)
        nextStateQval = None
        if bestActionForTheNextState is None:
            nextStateQval = 0
        else:
            nextStateQval = self.getQValue(nextState, bestActionForTheNextState)
        delta = reward + (self.discount * nextStateQval) - self.getQValue(state, action)
        #
        #delta = reward + (self.discount * self.computeValueFromQValues(nextState)) - self.getQValue(state, action)
        #
        featuresDict = self.featExtractor.getFeatures(state,action)
        for key_f in featuresDict:
            #self.weights[(state,action,key_f)] += self.alpha * delta * featuresDict[key_f]
            self.weights[key_f] += self.alpha * delta * featuresDict[key_f]
        # cornered case issue for question 4, HA3
        '''
        # Approximate TD-learning code below
        qvalue = 0
        nextStateQval = 0
        featuresDictS1 = self.featExtractor.getFeatures(state,action)
        for f in featuresDictS1:
            qvalue += self.weights[f] * featuresDictS1[f]
        #
        bestActionForTheNextState = self.computeActionFromQValues(nextState)
        if bestActionForTheNextState == None:
            nextStateQval = 0.0
        else:
            #bestActionForTheNextState is equal to None, Terminal state
            featuresDictS2 = self.featExtractor.getFeatures(nextState,bestActionForTheNextState)
            for f in featuresDictS2:
                nextStateQval += self.weights[f] * featuresDictS2[f]
        #
        if reward != -1:
            for f in featuresDictS1:
                self.EligTrVal[f] = 0
        #
        # TEST, task 4, HA4
        #
        bestActionForTheCurrentState = self.computeActionFromQValues(state)
        #
        # overwriting the nextStateQval with proposed modification
        featuresDictTest = self.featExtractor.getFeatures(nextState,action)
        nextStateQval = 0 # WHY would we do it here since it won't take any effect? Where should it actually be changed?
        for f in featuresDictTest:
            nextStateQval += self.weights[f] * featuresDictTest[f]
        #
        testNextStateQvalue = 0
        if bestActionForTheCurrentState is not None:
            testNextStateQvalue = self.getQValue(nextState, action)
        #
        delta = reward + self.discount * testNextStateQvalue - self.getQValue(state, action)
        #
        
        #
        # end of TEST, task 4, HA4
        #
        #delta = reward + self.discount * nextStateQval - qvalue
        #
        for f in featuresDictS1:
            self.EligTrVal[f] = self.lambdaVal * self.EligTrVal[f] + featuresDictS1[f]
            self.weights[f] += self.alpha * delta * self.EligTrVal[f]

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            print self.weights
            pass
