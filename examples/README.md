## Configuration Guide
This codebase uses `cfg.py`, `config.py`, and `env_config.py` for the configuration. 
### cfg.py
`cfg.py` contains two variables, `ENV` and `TASK`. They are used to select which environment and which task under that 
environment to run. For the iTHOR environment, only `ENV` is relevant and should be set to 'ai2thor'. 
For the Kuka environment, `ENV` should be set to 'arms' and the `TASK` variable should be set to 'fourInARow'.

### config.py
Each environment contains a `config.py` which has the settings for VAR, RL, and sound command.
For example, for the iTHOR environment, the file is located in `Envs/ai2thor`. 
The most important attributes are explained below and the full comment can be found in `config_commented.py`. It is 
recommended to look at `config_commented.py` before you start.
- render: if you want to visualize the environment, set it to True.
- pretextCollection: set it to True to collect triplets before training the VAR
- pretextTrain: set it to True to train the VAR.
- pretextDataDir: the path to your collected triplets.
- pretextNumEnvs: use this number of parallel environments to collect pretext data.
- pretextModelFineTune: set it to True if you collect additional data to fine tune a trained VAR.
- pretextModelSaveDir: the path to save the trained VAR.
- pretextModelLoadDir: the path to load the trained VAR for the RL training. 
- pretextCollectNum: for each class, collect this number of triplets. 
- pretextEpoch: train this number of epoch  for the VAR.
- RLManualControl: set it to True if you want to control the agent using keyboard. 
- RLTrain: set it to True if you want to perform the RL training. 
- RLModelSaveDir: the path to save the RL model
- RLModelFineTune: set it to True if you have a trained RL model and want to fine tune it.
- soundSource: a dictionary contains the settings for sound data.

The examples below show some commonly used settings.

- Visualize an environment and use keyboard to control the agent: `RLManualControl=True, RLTrain=False, render=True` and run `RL.py`
- Collect triplets automatically from the environment: `pretextCollection=True`, set `pretextCollectNum, pretextDataDir, soundSource`, and run `pretext.py`
- Train a VAR with collected triplets: `pretextTrain=True, pretextCollection=False, pretextModelFineTune=False`, set `pretextDataDir`, and run `pretext.py`
- Show embedding space of a trained VAR using the collected data: `pretextTrain=False, pretextCollection=False`, set `pretextDataDir, pretextModelLoadDir`, and run `pretext.py`
- Train an RL agent with a VAR: `RLTrain=True, RLManualControl=False, RLModelFineTune=False`, set `RLModelSaveDir, pretextModelLoadDir, soundSource`, and run `RL.py`
- Test an RL agent: `RLTrain=False, RLManualControl=False, RLModelFineTune=False`, set `RLModelLoadDir, pretextModelLoadDir, skillInfos, soundSource`, and run `RL.py`
- Manually collect triplets from the environment: `pretextTrain=False, pretextCollection=True, pretextManualControl=True, pretextManualCollect=True`, set `pretextDataDir`, and run `pretext.py`
- Show a real-time embedding space while controlling the agent with the keyboard: `RLManualControl=True, RLTrain=False, render=True, realTimeVec=True`, set `pretextDataDir, pretextModelLoadDir`and run `RL.py`
- Fine-tune a VAR with newly collected triplets: `pretextTrain=True, pretextCollection=False, pretextModelFineTune=True`, set `pretextDataDir, pretextModelLoadDir`, and run `pretext.py`
- Fine-tune an RL policy given an updated VAR: `RLTrain=True, RLManualControl=False, RLModelFineTune=True`, set `RLModelSaveDir, pretextModelLoadDir, soundSource`, and run `RL.py`


### env_config.py
This configuration file contains the settings for the environment. Usually, there is no need to 
change the settings in this file, unless you want to modify the environment. The keyboard control for the environment is contained in this file.