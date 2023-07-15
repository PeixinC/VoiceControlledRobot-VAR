from gym.envs.registration import register
import importlib
import os

ENV='ai2thor' # choose from ai2thor, arms
TASK='fourInARow' # 'fourInARow' for 'arms' only

class printColor:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def main_config():
    config=None
    if ENV == 'ai2thor':
        s = 'Envs.ai2thor.'+ 'config'
        config = getattr(importlib.import_module(s), 'AI2ThorConfig')()

        s = 'Envs.ai2thor.' + 'env_config'
        env_config = getattr(importlib.import_module(s), 'EnvConfig')
        config.get_env_config(env_config)

    elif ENV == 'arms':
        if TASK in ['fourInARow', ]:
            s='Envs.pybullet.arms.tasks.'+TASK+'.config'
            config=getattr(importlib.import_module(s), 'ArmConfig')()

            s='Envs.pybullet.arms.tasks.'+TASK+'.'+config.robotType+'.env_config'
            env_config = getattr(importlib.import_module(s), 'EnvConfig')

            config.get_env_config(env_config)

        else: raise NotImplementedError

    else: raise NotImplementedError


    return config

def gym_register(config):
    # register gym env

    env_folder = '' if ENV == 'ai2thor' else '.pybullet'

    if ENV=='arms':
        entry_point = 'Envs' + env_folder + '.' + ENV + '.' + 'tasks.'+TASK+ '.pretext_env_VAR' + ':PretextEnvVAR'
        register(
            id=ENV + '-pretext-v2',
            entry_point=entry_point
        )
        entry_point = 'Envs' + env_folder + '.' + ENV + '.' + 'tasks.'+TASK+ '.RL_env_VAR' + ':RLEnvVAR'
        register(
            id=ENV + '-RL-v2',
            entry_point=entry_point
        )

    else:
        entry_point = 'Envs' + env_folder + '.' + ENV + '.pretext_env_VAR'+ ':PretextEnvVAR'
        register(
            id= ENV + '-pretext-v2',
            entry_point=entry_point
        )
        entry_point = 'Envs' + env_folder + '.' + ENV + '.RL_env_VAR' + ':RLEnvVAR'
        register(
            id=ENV + '-RL-v2',
            entry_point=entry_point
        )

class configBase(object):
    def __init__(self):
        pass

    def print(self, txt, color):
        """
        print colored text
        :param txt: the string to be printed
        :param color: the color specified in printColor class
        :return: None
        """
        print(color+txt+printColor.ENDC)
    def get_env_config(self, config):
        config(self)

    def __setattr__(self, name, value):
        """
        Prevent reassignment of the same attribute
        :param name: name of the attribute
        :param value: value of the attribute
        :return:
        """
        if name in self.__dict__ and name!='taskNum':
            self.print("Reassignment of "+name+" to "+str(value), printColor.WARNING)
        self.__dict__[name] = value

    def cfg_check(self):
        # checking configuration and output errors or warnings
        flag=False
        if self.RLTrain and self.RLManualControl:
            raise Exception('self.RLTrain and self.RLManualControl cannot be both True')

        if 0 < self.episodeImgSaveInterval < 5:
            self.print("You may save the episode image too frequently", printColor.WARNING)

        if not flag:
            self.print("Configuration Check Passed!", printColor.OKGREEN)