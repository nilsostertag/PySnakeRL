import torch
import random
import numpy as np
from collections import deque
from snake_pygame.game import SnakeGameRL, Direction, Point, BLOCK_SIZE
from model import Linear_QNet, QTrainer
from plotter import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class Agent:
    
    def __init__(self):
        self.number_of_games = 0
        self.epsilon = 0 # control the randomness
        self.gamma = 0.9 # discount rate, gamma < 1
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        self.model = Linear_QNet(11, 256, 3)
        self.trainer = QTrainer(self.model, lr = LR, gamma=self.gamma)
        # todo: model, trainer

    def get_state(self, game):
        # get the snake's head from the game
        head = game.snake[0]

        # check the surrounding of the head in one unit radius
        point_l = Point(head.x - BLOCK_SIZE, head.y)
        point_r = Point(head.x + BLOCK_SIZE, head.y)
        point_u = Point(head.x, head.y - BLOCK_SIZE)
        point_d = Point(head.x, head.y + BLOCK_SIZE)

        # check if the current game direction equals
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Danger straight ahead
            (dir_r and game.is_collision(point_r)) or  # Look right, danger in right dir
            (dir_l and game.is_collision(point_l)) or  # Look left, danger in left dir
            (dir_u and game.is_collision(point_u)) or  # Look up, danger in up dir
            (dir_d and game.is_collision(point_d)),    # Look down, danger in down dir
                                                    # --> in front of head
            # Danger right
            (dir_u and game.is_collision(point_r)) or  # Look up, danger in right dir
            (dir_d and game.is_collision(point_l)) or  # Look down, danger in left dir
            (dir_l and game.is_collision(point_u)) or  # Look left, danger in up dir
            (dir_r and game.is_collision(point_d)),    # Look right, danger in down dir 
                                                    # --> right of head

            # Danger left
            (dir_d and game.is_collision(point_r)) or 
            (dir_u and game.is_collision(point_l)) or  
            (dir_r and game.is_collision(point_u)) or  
            (dir_l and game.is_collision(point_d)),    

            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # Food location
            game.food.x < game.head.x, # food left
            game.food.x > game.head.x, # food right
            game.food.y < game.head.y, # food up
            game.food.y > game.head.y  # food down 
        ]

        # numpy array of type 'int' to turn booleans into 0 or 1
        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, game_over):
        self.memory.append((state, action, reward, next_state, game_over)) # popleft if MAX_MEMORY is reached, double parenthesis for appending as a tuple

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            miniSample = random.sample(self.memory, BATCH_SIZE) # returns list of tuples
        else:
            miniSample = self.memory

        #for state, action, reward, nextState, gameOver in miniSample:
        #    self.trainer.trainStep(state, action, reward, nextState, gameOver)

        # using zip function for efficiency
        states, actions, rewards, next_states, game_overs = zip(*miniSample)
        self.trainer.train_step(states, actions, rewards, next_states, game_overs)

    def train_short_memory(self, state, action, reward, next_state, game_over):
        self.trainer.train_step(state, action, reward, next_state, game_over)

    def get_action(self, state):
        # random moves: tradeoff between exploration / exploitation
        self.epsilon = 80 - self.number_of_games
        final_move = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1
        
        return final_move

def train():
    scores = []
    mean_scores = []
    total_score = 0
    record_score = 0
    agent = Agent()
    game = SnakeGameRL()

    while True:
        # get the current state
        state_old = agent.get_state(game)    

        # get move
        final_move = agent.get_action(state_old)

        #perform move and get new state 
        reward, game_over, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # train the short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, game_over)

        # remember
        agent.remember(state_old, final_move, reward, state_new, game_over)

        if game_over:
            # train the long memory ("Experienced replay"), plot result
            game.reset()
            agent.number_of_games += 1
            agent.train_long_memory()

            if(score > record_score):
                record_score = score
                agent.model.save()

            print('Game: ', agent.number_of_games, '\nScore: ', score, '\nRecord: ', record_score)

            scores.append(score)
            total_score += score
            mean_score = total_score / agent.number_of_games
            mean_scores.append(mean_score)
            plot(scores, mean_scores)

if __name__ == '__main__':
    train()