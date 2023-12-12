import torch
import random
import numpy as np
from collections import deque
from snake_pygame.game import SnakeGameRL, Direction, Point, BLOCK_SIZE as gameUnit
from model import LinearQNet, QTrainer
from plotter import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class Agent:
    
    def __init__(self):
        self.numberOfGames = 0
        self.epsilon = 0 # control the randomness
        self.gamma = 0.9 # discount rate, gamma < 1
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        self.model = LinearQNet(11, 256, 3)
        self.trainer = QTrainer(self.model, lr = LR, gamma=self.gamma)
        # todo: model, trainer

    def getState(self, game):
        # get the snake's head from the game
        head = game.snake[0]

        # check the surrounding of the head in one unit radius
        pointL = Point(head.x - gameUnit, head.y)
        pointR = Point(head.x + gameUnit, head.y)
        pointU = Point(head.x, head.y + gameUnit)
        pointD = Point(head.x, head.y - gameUnit)

        # check if the current game direction equals
        dirL = game.direction == Direction.LEFT
        dirR = game.direction == Direction.RIGHT
        dirU = game.direction == Direction.UP
        dirD = game.direction == Direction.DOWN

        state = [
            # Danger straight ahead
            (dirR and game.isCollision(pointR)) or  # Look right, danger in right dir
            (dirL and game.isCollision(pointL)) or  # Look left, danger in left dir
            (dirU and game.isCollision(pointU)) or  # Look up, danger in up dir
            (dirD and game.isCollision(pointD)),    # Look down, danger in down dir
                                                    # --> in front of head
            # Danger right
            (dirU and game.isCollision(pointR)) or  # Look up, danger in right dir
            (dirD and game.isCollision(pointL)) or  # Look down, danger in left dir
            (dirL and game.isCollision(pointU)) or  # Look left, danger in up dir
            (dirR and game.isCollision(pointD)),    # Look right, danger in down dir 
                                                    # --> right of head

            # Danger left
            (dirD and game.isCollision(pointR)) or 
            (dirU and game.isCollision(pointL)) or  
            (dirR and game.isCollision(pointU)) or  
            (dirL and game.isCollision(pointD)),    

            # Move direction
            dirL,
            dirR,
            dirU,
            dirD,

            # Food location
            game.food.x < game.head.x, # food left
            game.food.x > game.head.x, # food right
            game.food.y < game.head.y, # food up
            game.food.y > game.head.y  # food down 
        ]

        # numpy array of type 'int' to turn booleans into 0 or 1
        return np.array(state, dtype=int)

    def remember(self, state, action, reward, nextState, gameOver):
        self.memory.append((state, action, reward, nextState, gameOver)) # popleft if MAX_MEMORY is reached, double parenthesis for appending as a tuple

    def trainLongMemory(self):
        if len(self.memory) > BATCH_SIZE:
            miniSample = random.sample(self.memory, BATCH_SIZE) # returns list of tuples
        else:
            miniSample = self.memory

        #for state, action, reward, nextState, gameOver in miniSample:
        #    self.trainer.trainStep(state, action, reward, nextState, gameOver)

        # using zip function for efficiency
        states, actions, rewards, nextStates, gameOvers = zip(*miniSample)
        self.trainer.trainStep(states, actions, rewards, nextStates, gameOvers)

    def trainShortMemory(self, state, action, reward, nextState, gameOver):
        self.trainer.trainStep(state, action, reward, nextState, gameOver)

    def getAction(self, state):
        # random moves: tradeoff between exploration / exploitation
        self.epsilon = 80 - self.numberOfGames
        finalMove = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            finalMove[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            finalMove[move] = 1
        
        return finalMove

def train():
    scores = []
    meanScores = []
    totalScore = 0
    recordScore = 0
    agent = Agent()
    game = SnakeGameRL()
    while True:
        # get the current state
        stateOld = agent.getState(game)    

        # get move
        finalMove = agent.getAction(stateOld)

        #perform move and get new state 
        reward, gameOver, score = game.playStep(finalMove)
        stateNew = agent.getState(game)

        # train the short memory
        agent.trainShortMemory(stateOld, finalMove, reward, stateNew, gameOver)

        # remember
        agent.remember(stateOld, finalMove, reward, stateNew, gameOver)

        if gameOver:
            # train the long memory ("Experienced replay"), plot result
            game.reset()
            agent.numberOfGames += 1
            agent.trainLongMemory()

            if(score > recordScore):
                recordScore = score
                agent.model.save()

            print('Game: ', agent.numberOfGames, 'Score: ', score, 'Record: ', recordScore)

            scores.append(score)
            totalScore += score
            meanScore = totalScore / agent.numberOfGames
            meanScores.append(meanScore)
            plot(scores, meanScores)

if __name__ == '__main__':
    train()