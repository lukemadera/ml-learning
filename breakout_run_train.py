import gym
import logging
import numpy as np
import torch
import time

import breakout_ai_a2c as ai_a2c
import date_time
import number
from subproc_vec_env import SubprocVecEnv
from atari_wrappers import make_atari, wrap_deepmind, Monitor

def updateState(obs, state, nc):
    # Do frame-stacking here instead of the FrameStack wrapper to reduce IPC overhead
    state = np.roll(state, shift=-nc, axis=3)
    state[:, :, :, -nc:] = obs
    return state

def runTrain(gymId='BreakoutNoFrameskip-v4', numEnvs=16, seed=0, filePathBrain='training/breakout-v1.pth',
    numSteps=5, numBatches=20000, outputBatchInterval=1000, joinEnvs=1, epsilon=0.00001):
    def make_env(rank):
        def _thunk():
            env = make_atari(gymId)
            env.seed(seed + rank)
            gym.logger.setLevel(logging.WARN)
            env = wrap_deepmind(env)

            # wrap the env one more time for getting total reward
            env = Monitor(env, rank)
            return env
        return _thunk

    print ('training starting', numBatches, outputBatchInterval,
        'epsilon', epsilon)
    env = SubprocVecEnv([make_env(i) for i in range(numEnvs)])

    numActions = env.action_space.n

    torchDevice = 'cpu'
    if torch.cuda.is_available():
        torchDevice = 'cuda'
    agent = ai_a2c.A2C(numActions, device=torchDevice)
    if filePathBrain:
        agent.load(filePath=filePathBrain)

    timingStart = date_time.now()
    batchCount = 0

    states, actions, rewards, dones, values = [], [], [], [], []
    for ii in range(numEnvs):
        states.append([])
        actions.append([])
        rewards.append([])
        dones.append([])
        values.append([])

    # Set first state.
    # Environment returns 1 frame, but we want multiple, so we stack the new
    # state on top of the past ones.
    nh, nw, nc = env.observation_space.shape
    nstack = 4
    batchStateShape = (numEnvs * numSteps, nh, nw, nc * nstack)
    emptyState = np.zeros((numEnvs, nh, nw, nc * nstack), dtype=np.uint8)
    obs = env.reset()
    # states = updateState(obs, emptyState, nc)
    lastStates = updateState(obs, emptyState, nc)
    lastDones = [False for _ in range(numEnvs)]

    totalRewards = []
    realTotalRewards = []
    # All actions are always valid.
    validActions = [0,1,2,3]

    while batchCount < numBatches:
        states, actions, rewards, dones, values = [], [], [], [], []
        stepCount = 0
        while stepCount < numSteps:
            actionsStep, valuesStep = agent.selectActions(lastStates, validActions=validActions, randomRatio=epsilon)
            # print ('actionsStep', actionsStep)
            states.append(np.copy(lastStates))
            actions.append(actionsStep)
            values.append(valuesStep)
            if stepCount > 0:
                dones.append(lastDones)

            # Input the action (run a step) for all environments.
            statesStep, rewardsStep, donesStep, infosStep = env.step(actionsStep)

            # Update state for any dones.
            for n, done in enumerate(donesStep):
                if done:
                    lastStates[n] = lastStates[n] * 0
            lastStates = updateState(obs, lastStates, nc)

            # Update rewards for logging / tracking.
            for done, info in zip(donesStep, infosStep):
                if done:
                    totalRewards.append(info['reward'])
                    if info['total_reward'] != -1:
                        realTotalRewards.append(info['total_reward'])

            lastDones = donesStep
            rewards.append(rewardsStep)

            stepCount += 1

        # Dones is one off, so add the last one.
        dones.append(lastDones)

        # discount/bootstrap off value fn
        # lastValues = self.agent.value(lastStates).tolist()
        # Can skip this as it is done in the learn function with calcActualStateValues?

        # Join all (combine batches and steps).
        states = np.asarray(states, dtype='float32').swapaxes(1, 0).reshape(batchStateShape)
        actions = np.asarray(actions).swapaxes(1, 0).flatten()
        rewards = np.asarray(rewards).swapaxes(1, 0).flatten()
        dones = np.asarray(dones).swapaxes(1, 0).flatten()
        values = np.asarray(values).swapaxes(1, 0).flatten()
        agent.learn(states, actions, rewards, dones, values)

        batchCount += 1

        if batchCount % outputBatchInterval == 0:
            runTime = date_time.diff(date_time.now(), timingStart, 'minutes')
            totalSteps = batchCount * numSteps
            runTimePerStep = runTime / totalSteps
            runTimePerStepUnit = 'minutes'
            if runTimePerStep < 0.02:
                runTimePerStep *= 60
                runTimePerStepUnit = 'seconds'
            print (batchCount, numBatches, '(batch done)',
                number.toFixed(runTime), 'run time minutes,', totalSteps,
                'steps,', number.toFixed(runTimePerStep), runTimePerStepUnit, 'per step')

            r = totalRewards[-100:] # get last 100
            tr = realTotalRewards[-100:]
            if len(r) == 100:
                print("avg reward (last 100):", np.mean(r))
            if len(tr) == 100:
                print("avg total reward (last 100):", np.mean(tr))
                print("max (last 100):", np.max(tr))

            # Only save periodically as well.
            if filePathBrain:
                agent.save(filePathBrain)

    env.close()

    if filePathBrain:
        agent.save(filePathBrain)

    runTime = date_time.diff(date_time.now(), timingStart, 'minutes')
    totalSteps = numBatches * numSteps
    runTimePerStep = runTime / totalSteps
    runTimePerStepUnit = 'minutes'
    if runTimePerStep < 0.02:
        runTimePerStep *= 60
        runTimePerStepUnit = 'seconds'
    print ('training done:', number.toFixed(runTime), 'run time minutes,', totalSteps,
        'steps,', number.toFixed(runTimePerStep), runTimePerStepUnit, 'per step')

    return None

runTrain(filePathBrain='training/breakout-v1-2.pth', epsilon=0.0001)
