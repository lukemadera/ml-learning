import numpy as np
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd

# Implementing a function to make sure the models share the same gradient
# def ensure_shared_grads(model, shared_model):
#     for param, shared_param in zip(model.parameters(), shared_model.parameters()):
#         if shared_param.grad is not None:
#             return
#         shared_param._grad = param.grad

class ActorCritic(nn.Module):
    def __init__(self, numActions, numInputs=84):
        super(ActorCritic, self).__init__()

        # self.conv1 = nn.Conv2d(numInputs, 32, kernel_size=8, stride=4)
        # self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        # self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        self.conv1 = nn.Conv2d(numInputs, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)

        self.linear1 = nn.Linear(192, 512)

        self.actor = nn.Linear(512, numActions)
        self.critic = nn.Linear(512, 1)
    
    # In a PyTorch model, you only have to define the forward pass.
    # PyTorch computes the backwards pass for you!
    def forward(self, x):
        # Normalize image pixels (from rgb 0 to 255) to between 0 and 1.
        x = x / 255.
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.linear1(x))
        return x
    
    # Only the Actor head
    def get_action_probs(self, x):
        x = self(x)
        actionProbs = F.softmax(self.actor(x), dim=1)
        actionProbs = torch.clamp(actionProbs, 0.0001, 0.9999)
        return actionProbs
    
    # Only the Critic head
    def getStateValue(self, x):
        x = self(x)
        stateValue = self.critic(x)
        return stateValue
    
    # Both heads
    def evaluate_actions(self, x):
        x = self(x)
        actionProbs = F.softmax(self.actor(x), dim=1)
        actionProbs = torch.clamp(actionProbs, 0.0001, 0.9999)
        stateValues = self.critic(x)
        return actionProbs, stateValues

class A2C():

    def __init__(self, numActions, gamma=None, learningRate=None, maxGradNorm=0.5,
        entropyCoefficient=0.01, valueLossFactor=0.5, sharedModel=None,
        sharedOptimizer=None, device='cpu'):
        self.gamma = gamma if gamma is not None else 0.99
        self.learningRate = learningRate if learningRate is not None else 0.0007
        self.maxGradNorm = maxGradNorm
        self.entropyCoefficient = entropyCoefficient
        self.valueLossFactor = valueLossFactor
        self.model = ActorCritic(numActions).to(device=device)
        self.sharedModel = sharedModel
        self.optimizer = sharedOptimizer if sharedOptimizer is not None else \
            optim.Adam(self.model.parameters(), lr=self.learningRate)
        self.device = device
        print ('A2C hyperparameters',
            'learningRate', self.learningRate,
            'gamma', self.gamma,
            'entropyCoefficient', self.entropyCoefficient,
            'valueLossFactor', self.valueLossFactor,
            'maxGradNorm', self.maxGradNorm)

    def save(self, filePath='training-runs/a2c.pth'):
        torch.save({'state_dict': self.model.state_dict(),
                    'optimizer' : self.optimizer.state_dict(),
                   }, filePath)
        print("=> saved checkpoint... ", filePath)
    
    def load(self, filePath='training-runs/a2c.pth'):
        if os.path.isfile(filePath):
            print("=> loading checkpoint... ", filePath)
            checkpoint = torch.load(filePath)
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("done!")
        else:
            print("no checkpoint found...", filePath)

    # def syncSharedModel(self):
    #     if self.sharedModel is not None:
    #         # Synchronizing with the shared model
    #         self.model.load_state_dict(self.sharedModel.state_dict())

    def getValues(self, state):
        stateTensor = torch.tensor(state, dtype=torch.float32, device=self.device)
        return self.model.get_action_probs(stateTensor)

    def pickAction(self, bestAction, validActions=None, randomRatio=-1):
        action = bestAction
        if randomRatio >= 0 and validActions is not None:
            randNum = random.uniform(0, 1)
            if randNum < randomRatio:
                action = np.random.choice(validActions)
                # print ('random action')
        # action = actionProbs.multinomial(num_samples=1)
        # action = action[0,0].tolist()

        if validActions is not None and action not in validActions:
            action = np.random.choice(validActions)
        return action

    def selectActions(self, states, validActions=None, randomRatio=-1):
        statesTensor = torch.tensor(states, dtype=torch.float32, device=self.device)
        actionProbs, stateValues = self.model.evaluate_actions(statesTensor)

        actions = []
        for item in actionProbs:
            bestAction = item.max(0)[1].tolist()
            action = self.pickAction(bestAction, validActions, randomRatio)
            actions.append(action)
        return actions, stateValues.tolist()

    def selectAction(self, state, validActions=None, randomRatio=-1):
        # Need to add dimension to simulate stack of states, even though just have one.
        stateTensor = torch.tensor(state, dtype=torch.float32, device=self.device)
        actionProbs, stateValues = self.model.evaluate_actions(stateTensor)

        _, bestAction = actionProbs.max(maxIndex)
        bestAction = bestAction[0].tolist()
        action = self.pickAction(bestAction, validActions, randomRatio)

        return action, stateValues

    def calcActualStateValues(self, rewards, dones, statesTensor):
        rewards = rewards.tolist()
        dones = dones.tolist()
        # R is the cumulative reward.
        R = []
        rewards.reverse()

        if dones[-1]:
        # if 0:
            nextReturn = 0
        else:
            stateTensor = statesTensor[-1].unsqueeze(0)
            nextReturn = self.model.getStateValue(stateTensor)[0][0].tolist()
        
        # Backup from last state to calculate "true" returns for each state in the set
        R.append(nextReturn)
        dones.reverse()
        for r in range(1, len(rewards)):
            if dones[r]:
            # if 0:
                thisReturn = 0
            else:
                thisReturn = rewards[r] + nextReturn * self.gamma
                # print ('thisReturn', thisReturn, rewards[r], nextReturn, self.gamma, rewards, r)
            R.append(thisReturn)
            nextReturn = thisReturn

        R.reverse()
        # print ('rewards', rewards)
        stateValuesActual = torch.tensor(R, dtype=torch.float32, device=self.device).unsqueeze(1)
        # print ('stateValuesActual', stateValuesActual)
        # print ('R', R)
        
        return stateValuesActual

    def learn(self, states, actions, rewards, dones, values=None):
        statesTensor = torch.tensor(states, dtype=torch.float32, device=self.device)
        # s = torch.tensor(states, dtype=torch.float32, device=self.device)
        # Need to convert from array of tensors to tensor of tensors.
        # actionProbs, stateValuesEst = self.model.evaluate_actions(torch.cat(statesTensor, 0))
        actionProbs, stateValuesEst = self.model.evaluate_actions(statesTensor)
        # print ('actionProbs', actionProbs)
        # print ('stateValuesEst', stateValuesEst)

        actionLogProbs = actionProbs.log()
        # print ('actionProbs', actionProbs)
        # print ('actionLogProbs', actionLogProbs)

        a = torch.tensor(actions, dtype=torch.int64, device=self.device).view(-1,1)
        chosenActionLogProbs = actionLogProbs.gather(1, a)
        # print ('chosenActionLogProbs', chosenActionLogProbs)

        versionToUse = 'v1'
        # v1 - original
        if versionToUse == 'v1':
            # Calculating the actual values.
            stateValuesActual = self.calcActualStateValues(rewards, dones, statesTensor)
            # print ('stateValuesActual', stateValuesActual)

            # This is also the TD (Temporal Difference) error
            advantages = stateValuesActual - stateValuesEst
            # print ('advantages', advantages)

            valueLoss = advantages.pow(2).mean()
            # print ('value_loss', value_loss)

            entropy = (actionProbs * actionLogProbs).sum(1).mean()
            # print ('entropy', entropy, actionProbs, actionLogProbs)
            actionGain = (chosenActionLogProbs * advantages).mean()
            # print ('actiongain', actionGain)
            totalLoss = self.valueLossFactor * valueLoss - \
                actionGain - self.entropyCoefficient * entropy
            # print ('totalLoss', totalLoss, valueLoss, actionGain)

        # v2 - http://steven-anker.nl/blog/?p=184
        if versionToUse == 'v2':
            R = 0
            if not dones[-1]:
                stateTensor = statesTensor[-1]
                R = self.model.getStateValue(stateTensor)[0][0].tolist()
             
            n = len(statesTensor)
            VF = stateValuesEst
            RW = np.zeros(n)
            ADV = np.zeros(n)
            A = np.array(actions)
             
            for i in range(n - 1, -1, -1):
                R = rewards[i] + self.gamma * R
                RW[i] = R
                ADV[i] = R - VF[i]
            advantages = torch.from_numpy(ADV, device=self.device)

            # rewardsTensor = []
            # for reward in rewards:
            #     print (reward, torch.tensor([reward], device=self.device))
            #     rewardsTensor.append(torch.tensor(reward, device=self.device))
            rewardsTensor = list(map(lambda x: torch.tensor([x], device=self.device), rewards))
            rewardsTensor = torch.cat(rewardsTensor, 0)
            valueLoss = 0.5 * (stateValuesEst - rewardsTensor).pow(2).mean()
            # valueLoss = 0.5 * (stateValuesEst - torch.from_numpy(RW, device=self.device)).pow(2).mean()

            actionOneHot = chosenActionLogProbs #Is this correct??
            negLogPolicy = -1 * actionLogProbs
            # Only the output related to the action needs to be adjusted, since we only know the result of that action.
            # By multiplying the negative log of the policy output with the one hot encoded vectors, we force all outputs
            # other than the one of the action to zero.
            policyLoss = ((negLogPolicy * actionOneHot).sum(1) * advantages.float()).mean()
            entropy = (actionProbs * negLogPolicy).sum(1).mean()
            # Training works best if the value loss has less influence than the policy loss, so reduce value loss by a factor.
            # Optimizing with this loss function could result in converging too quickly to a sub optimal solution.
            # I.e. the probability of a single action is significant higher than any other, causing it to always be chosen.
            # To prevent this we add a penalty on having a high entropy.
            totalLoss = self.valueLossFactor * valueLoss + policyLoss - self.entropyCoefficient * entropy


        self.optimizer.zero_grad()
        totalLoss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), self.maxGradNorm)
        # if self.sharedModel is not None:
        #     ensure_shared_grads(self.model, self.sharedModel)
        self.optimizer.step()
