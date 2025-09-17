from mainle.agents import Agent
from mainle.chat import ChatEngine


class LlmAgent(Agent):
    def __init__(self, chat_engine: ChatEngine):
        self.chat_engine = chat_engine

    def save_history(self, json_filename: str):
        self.chat_engine.save_history(json_filename)
