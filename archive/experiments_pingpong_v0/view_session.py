"""Episode lifecycle for the interactive demo; separate from learned inference."""

class InferenceSession:
    def __init__(self, task, seed=81000):
        self.task = task
        self.seed = seed
        self.episode = 0
        self.last_restart_reason = ""
        self.restart()

    def restart(self, reason=""):
        # Task.reset clears physical state, tracker, target, history and delay buffers.
        # Reuse the same initial condition so consecutive trials remain comparable.
        self.observation = self.task.reset(self.seed)
        self.episode += 1
        self.last_restart_reason = reason
        return self.observation

    def advance(self, action=None, wheel_override=None):
        observation, _, _, failure = self.task.step(action, wheel_override=wheel_override)
        if failure:
            reason = getattr(self.task, "failure_reason", "") or "Sensor or motor limit failure"
            self.restart(reason)
            return True
        self.observation = observation
        return False
