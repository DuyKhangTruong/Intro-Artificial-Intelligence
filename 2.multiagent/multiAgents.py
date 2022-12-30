# multiAgents.py
# --------------
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


# from tables import FloatAtom
from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        if successorGameState.isWin():
            return 99999
        
        # We check the closest food from the Pacman
        closestFoodPos = min([manhattanDistance(newPos,foodPos) for foodPos in newFood.asList()])
        # Checking when the ghosts are not scared and the Pacman's position is too close to the ghosts, we consider
        # Pacman is lost
        for ghost in newGhostStates:
            if ghost.scaredTimer == 0 and manhattanDistance(newPos,ghost.getPosition()) <= 1:
                return -99999
        # Return the score based on the distance to the food position, the closer the pacman is the higher score
        return successorGameState.getScore() + (1 / closestFoodPos)

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 #Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        agentIndex = 0 #Pacman
        depth = 0
        # valueFunction returns tuple(value,action)
        action = self.valueFunction(gameState,agentIndex,depth)[1]
        return action
        util.raiseNotDefined()
    
    def valueFunction(self,gameState,agentIndex,depth):
        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return (self.evaluationFunction(gameState),"")
        if agentIndex == 0:
            return self.MaxAgent(gameState,agentIndex,depth)
        else:
            return self.MinAgent(gameState,agentIndex,depth)
        
    
    def MaxAgent(self,gameState,agentIndex,depth):
        legalActions = gameState.getLegalActions(agentIndex)
        maxValue = float("-inf")
        maxAction = ""
        for action in legalActions:
            successor = gameState.generateSuccessor(agentIndex,action)
            # We do not  need to check agent index for maxAgent because this step is
            # ensured with the MinAgent below
            currentValue = self.valueFunction(successor,agentIndex+1,depth)[0]
            if currentValue > maxValue:
                maxValue = currentValue
                maxAction = action
        return (maxValue,maxAction)
        
            
            
    def MinAgent(self,gameState,agentIndex,depth):
        legalActions = gameState.getLegalActions(agentIndex)
        minValue = float("inf")
        minAction = ""
        for action in legalActions:
            successor = gameState.generateSuccessor(agentIndex,action)
            # We check the index of agent. if we already switched to every agents: Pacman, ghosts
            # we will reset the index to 0 which is the pacman and increase the depth of the search tree
            # If not, we just need to increase the index of agent 
            if agentIndex + 1 >= gameState.getNumAgents():
                currentValue = self.valueFunction(successor,0,depth+1)[0]
            else:
                currentValue = self.valueFunction(successor,agentIndex+1,depth)[0]
            if currentValue < minValue:
                minValue = currentValue
                minAction = action
        return (minValue,minAction)
       

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        agentIndex = 0 #Pacman
        depth = 0
        alpha = float("-inf")
        beta = float("inf")
        action = self.alphaBetaFunction(gameState,agentIndex,depth,alpha,beta)[1]
        return action
        util.raiseNotDefined()
    
    def alphaBetaFunction(self,gameState,agentIndex,depth,alpha,beta):
        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return (self.evaluationFunction(gameState),"")
        if agentIndex == 0:
            return self.MaxAgent(gameState,agentIndex,depth,alpha,beta)
        else:
            return self.MinAgent(gameState,agentIndex,depth,alpha,beta)
        
    
    def MaxAgent(self,gameState,agentIndex,depth,alpha,beta):
        legalActions = gameState.getLegalActions(agentIndex)
        maxValue = float("-inf")
        maxAction = ""
        for action in legalActions:
            successor = gameState.generateSuccessor(agentIndex,action)
            currentValue = self.alphaBetaFunction(successor,agentIndex+1,depth,alpha,beta)[0]
            if currentValue > maxValue:
                maxValue = currentValue
                maxAction = action
            #If current max value is greather than beta, we break the loop
            if maxValue > beta:
                break
            alpha = max(alpha,maxValue)    
        return (maxValue,maxAction)
    
            
            
    def MinAgent(self,gameState,agentIndex,depth,alpha,beta):
        legalActions = gameState.getLegalActions(agentIndex)
        minValue = float("inf")
        minAction = ""
        for action in legalActions:
            successor = gameState.generateSuccessor(agentIndex,action)
            if agentIndex + 1 >= gameState.getNumAgents():
                currentValue = self.alphaBetaFunction(successor,0,depth+1,alpha,beta)[0]
            else:
                currentValue = self.alphaBetaFunction(successor,agentIndex+1,depth,alpha,beta)[0]
            if currentValue < minValue:
                minValue = currentValue
                minAction = action
            beta = min(beta, minValue)
            #If current min value is less than alpha, we break the loop
            if minValue < alpha:
                break
            beta = min(beta,minValue)
        return (minValue,minAction)

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        agentIndex = 0 #Pacman
        depth = 0
        action = self.valueFunction(gameState,agentIndex,depth)[1]
        return action
        util.raiseNotDefined()
        
    def valueFunction(self,gameState,agentIndex,depth):
        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return (self.evaluationFunction(gameState),"")
        if agentIndex == 0:
            return self.MaxAgent(gameState,agentIndex,depth)
        else:
            return self.expectiMaxAgent(gameState,agentIndex,depth)
        
    
    def MaxAgent(self,gameState,agentIndex,depth):
        legalActions = gameState.getLegalActions(agentIndex)
        maxValue = float("-inf")
        maxAction = ""
        for action in legalActions:
            successor = gameState.generateSuccessor(agentIndex,action)
            currentValue = self.valueFunction(successor,agentIndex+1,depth)[0]
            if currentValue > maxValue:
                maxValue = currentValue
                maxAction = action
        return (maxValue,maxAction)
        
    def expectiMaxAgent(self,gameState,agentIndex,depth):
        legalActions = gameState.getLegalActions(agentIndex)
        expectValue  = 0
        expectAction = ""
        # We could calculate the probability value by doing this formula 1 / (number of legal actions)
        probabilityValue = 1 / len(legalActions)
        for action in legalActions:
            successor = gameState.generateSuccessor(agentIndex,action)
            if agentIndex + 1 >= gameState.getNumAgents():
                currentValue,currentAction = self.valueFunction(successor,0,depth+1)
            else:
                currentValue,currentAction = self.valueFunction(successor,agentIndex + 1, depth)
            expectValue += (probabilityValue*currentValue)
            expectAction = currentAction
        return (expectValue,expectAction)

