from abc import abstractmethod

from mainle import agents
from mainle import chat


class Parser(agents.Agent):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def _response_to_dict(self, response: str) -> dict:
        pass

    @abstractmethod
    def execute(self):
        pass

    @abstractmethod
    def history(self):
        pass


class LlmParser(Parser):
    def __init__(self, engine: chat.ChatEngine):
        super().__init__()
        self.chat_engine = engine

    def _welcome_message(self) -> str:
        return "Welcome to the LLM Parser!"

    @abstractmethod
    def _response_to_dict(self, response: str) -> dict:
        pass

    def execute(self):
        response = self._welcome_message()
        parsed_response = self._response_to_dict(response)

        while not parsed_response:
            print(f"\n>> [{self.chat_engine}]: {response}")
            prompt = input("\n>> [User]: ")

            response = self.chat_engine.chat(prompt)
            parsed_response = self._response_to_dict(response)

        return parsed_response

    def history(self):
        return self.chat_engine.history()

    def save_history(self, json_filename: str, include_system_prompt: bool = False):
        self.chat_engine.save_history(json_filename, include_system_prompt=include_system_prompt)
