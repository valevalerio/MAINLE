import logging

import ollama

from mainle import chat


# TODO: add support to Llama token count
def count_tokens(prompt: str, model_name: str):
    return 0


class LlamaResponse(chat.Response):
    def __init__(self, response):
        self.response = response

    def __str__(self):
        return self.response.message.content

    def __repr__(self):
        return self.__str__()

    def raw_response(self):
        return self.response

    def to_message(self):
        message = chat.Message(
            role=self.response.message.role,
            model=self.response.model,
            content=self.response.message.content,
            tokens=self.response.eval_count,
        )
        return message


class Llama(chat.ChatEngine):
    def __init__(self, model, api_key=None, system_prompt="You are a helpful assistant.", tools=None):
        self.client = ollama.Client()
        self.model_name = model
        self.system_prompt = system_prompt
        self.tools = tools
        self._history = [
            chat.Message(
                role="system",
                model=model,
                content=self.system_prompt,
                tokens=count_tokens(self.system_prompt, model),
            )
        ]
        logging.info(f"Initialized Llama model: {model}")

    def parse_message(self, message: chat.Message):
        return [{"role": message["role"], "content": message["content"]}]

    def chat(self, prompt):
        if prompt is None:
            self._history.append(chat.Message(content="", tokens=0))
        elif isinstance(prompt, str):
            num_tokens = count_tokens(prompt, self.model_name)
            self._history.append(chat.Message(content=str(prompt), tokens=num_tokens))
        elif isinstance(prompt, chat.Message):
            prompt.tokens = count_tokens(prompt["content"], self.model_name)
            self._history.append(prompt)
        elif isinstance(prompt, list):
            for message in prompt:
                message.tokens = count_tokens(message["content"], self.model_name)
                self._history.append(message)
        else:
            raise TypeError(f"Unsupported prompt type: {type(prompt)}")

        return self.query()

    def query(self):
        messages = self.parse_history()

        response = self.client.chat(
            model=self.model_name,
            messages=messages,
            tools=self.tools,
        )
        response = LlamaResponse(response)

        self._history.append(response.to_message())

        return response

    def __repr__(self):
        return self.model_name
