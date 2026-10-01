# Agent interface. Each agent reads an image + a clinical note and returns P(anomaly) in [0, 1].
# Plug your existing multi-agent anomaly detection system in here.


class Agent:
    name = "base"

    def predict(self, image, note):
        raise NotImplementedError


class VLMAgent(Agent):
    # template: ask a VLM for a probability. Implement `_ask` with your model / API of choice.
    PROMPT = ("You are a radiology assistant. Given the image and note, estimate the probability (0-1) "
              "that an anomaly is present. Reply with only a number.\nNote: {note}")

    def __init__(self, name="vlm"):
        self.name = name

    def _ask(self, image, prompt):
        raise NotImplementedError

    def predict(self, image, note):
        return float(self._ask(image, self.PROMPT.format(note=note)))
