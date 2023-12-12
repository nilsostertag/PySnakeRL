import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

class LinearQNet(nn.Module):
    
    def __init__(self, inputSize, hiddenSize, outputSize):
        super().__init__()
        self.linear1 = nn.Linear(inputSize, hiddenSize)
        self.linear2 = nn.Linear(hiddenSize, outputSize)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x
    
    def save(self, fileName='model.pth'):
        modelFolderPath = './model'
        if not os.path.exists(modelFolderPath):
            os.makedirs(modelFolderPath)
        
        fileName = os.path.join(modelFolderPath, fileName)
        torch.save(self.state_dict(), fileName)

class QTrainer:
    
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def trainStep(self, state, action, reward, nextState, gameOver):
        state = torch.tensor(state, dtype=torch.float)
        nextState = torch.tensor(nextState, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)

        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            nextState = torch.unsqueeze(nextState, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            gameOver = (gameOver, )

        # 1: predicted Q values with current state
        pred = self.model(state)

        # 2: Q_new = r + y * max(next predicted Q value) -> only do this if not done
        target = pred.clone()
        for index in range(len(gameOver)):
            Q_new = reward[index]
            if not gameOver[index]:
                Q_new = reward[index] + self.gamma * torch.max(self.model(nextState[index]))

            target[index][torch.argmax(action).item()] = Q_new

        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()

        self.optimizer.step()