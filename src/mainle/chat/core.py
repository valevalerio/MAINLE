from abc import ABC, abstractmethod

import json
from datetime import datetime


class Message(dict):
    def __init__(self, content, role="user", model="user", tokens=0):
        dict.__init__(
            self,
            role = role,
            model = model,
            content = content,
            tokens = tokens,
            date = datetime.now().strftime("%Y-%m-%d-%H:%M:%S"),
        )

    def __repr__(self):
        return f"{self['date']} >> [{self['model']}]: {self['content']}"

    def __str__(self):
        return self.__repr__()

class Response(ABC):
    @abstractmethod
    def __repr__(self):
        pass

    @abstractmethod
    def to_message(self) -> Message:
        pass

class ChatEngine(ABC):
    @abstractmethod
    def chat(self, prompt: str) -> Response:
        pass

    @abstractmethod
    def parse_message(self, message: Message) -> list:
        pass

    def history(self, include_system_prompt: bool = False) -> list:
        if include_system_prompt:
            return self._history
        else:
            return self._history[1:]

    def parse_history(self) -> list:
        prompt = []

        for message in self._history:
            prompt.extend(self.parse_message(message))

        return prompt

    def save_history(self, json_filename: str, include_system_prompt: bool = False) -> None:
        with open(json_filename, "w") as f:
            json.dump(self.history(include_system_prompt=include_system_prompt), f, indent=4)
