import sys, os
sys.path.append(os.path.dirname(__file__))

import time



class SingleRobotEmptyScene(object):
    "A base class for single agent scenes"

    def __init__(self, bulletClient, gravity, timestep, frame_skip,render):

        self.timestep = timestep
        self.frame_skip = frame_skip
        self.p=bulletClient
        self.render=render
        self.dt = self.timestep * self.frame_skip
        self.cpp_world = World(bulletClient,gravity, timestep, frame_skip, render)

    def episode_restart(self):
        self.cpp_world.clean_everything()

    def global_step(self):
        self.cpp_world.step()


class World:

    def __init__(self, bulletClient,gravity, timestep, frame_skip,render):
        self.gravity = gravity
        self.timestep = timestep
        self.frame_skip = frame_skip
        self.p = bulletClient
        self.clean_everything()

        self.render=render


    def clean_everything(self):
        self.p.setGravity(self.gravity[0],self.gravity[1],self.gravity[2])

        self.p.setPhysicsEngineParameter(fixedTimeStep=self.timestep*self.frame_skip, numSolverIterations=30,
                                    numSubSteps=(self.frame_skip-1))


    def step(self):
        self.p.stepSimulation()
        if self.render:
            time.sleep(self.timestep*self.frame_skip)





