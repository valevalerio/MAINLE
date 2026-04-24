import ast
import re
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

    def _response_to_payload_dict(self, response: str):
        response_str = str(response)
        parts = response_str.split("```")
        payload = parts[1] if len(parts) >= 3 else parts[0]

        if payload and not payload.startswith("{"):
            start_index = payload.find("{")
            if start_index == -1:
                return None
            payload = payload[start_index:]

        if "{" not in payload or "}" not in payload:
            return None

        payload = payload.split("{")[1].split("}")[0]
        payload = "{" + payload + "}"

        try:
            return ast.literal_eval(payload)
        except (SyntaxError, ValueError):
            return None

    def _normalize_schema_name(self, value):
        return re.sub(r"[^a-z0-9]+", "", str(value).lower())

    def _response_to_exact_schema_dict(self, response: str, expected_fields: list[str], aliases: dict[str, list[str] | str] | None = None):
        parsed_response = self._response_to_payload_dict(response)
        if not parsed_response:
            return None

        parsed_lookup = {}
        for key, value in parsed_response.items():
            normalized_key = self._normalize_schema_name(key)
            if normalized_key in {self._normalize_schema_name("class"), self._normalize_schema_name("class_")}:
                continue
            parsed_lookup[normalized_key] = value

        resolved = {}
        for field_name in expected_fields:
            candidate_names = [field_name]
            if aliases and field_name in aliases:
                alias_values = aliases[field_name]
                if isinstance(alias_values, str):
                    alias_values = [alias_values]
                candidate_names.extend(alias_values)

            found_value = None
            for candidate_name in candidate_names:
                normalized_candidate = self._normalize_schema_name(candidate_name)
                if normalized_candidate in parsed_lookup:
                    found_value = parsed_lookup[normalized_candidate]
                    break

            if found_value is None:
                return None

            resolved[field_name] = found_value

        return resolved

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
