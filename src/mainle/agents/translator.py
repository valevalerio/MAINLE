from abc import abstractmethod

from mainle import agents
from mainle import chat


class Translator(agents.Agent):
    @abstractmethod
    def execute(self, prompt: str) -> str:
        pass

    @abstractmethod
    def history(self):
        pass


class LlmTranslator(Translator):
    def __init__(self, chat_engine: chat.ChatEngine):
        self.chat_engine = chat_engine

    def execute(self, prompt: str) -> str:
        raw_explanation = str(self.chat_engine.chat(prompt))
        return raw_explanation

    def history(self):
        return self.chat.history()

    def save_history(self, json_filename: str, include_system_prompt: bool = False):
        self.chat_engine.save_history(json_filename, include_system_prompt=include_system_prompt)
