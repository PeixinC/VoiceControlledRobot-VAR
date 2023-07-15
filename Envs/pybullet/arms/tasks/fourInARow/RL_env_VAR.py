from Envs.pybullet.arms.tasks.fourInARow.fourInARow import FourInARow
try: import sounddevice as sd
except: pass

class RLEnvVAR(FourInARow):
	def __init__(self):
		FourInARow.__init__(self)






