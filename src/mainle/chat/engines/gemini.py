import logging

from google import genai

from mainle import chat


# TODO: add support to Gemini token count
def count_tokens(prompt: str, model_name: str):
    return 0


class GeminiResponse(chat.Response):
    def __init__(self, response, model_name):
        self.response = response
        self.model_name = model_name

    def __str__(self):
        return self.response.text

    def __repr__(self):
        return self.__str__()

    def raw_response(self):
        return self.response

    def to_message(self):
        message = chat.Message(
            role="assistant",
            model=self.model_name,
            content=self.response.text,
            tokens=self.response.usage_metadata.candidates_token_count,
        )
        return message


class Gemini(chat.ChatEngine):
    def __init__(self, model, api_key: str = None, system_prompt: str = "You are a helpful assistant."):
        genai.configure(api_key=api_key)
        self.model_name = model
        self.system_prompt = system_prompt
        self.model = genai.GenerativeModel(model_name=model, system_instruction=system_prompt)
        self._history = [
            chat.Message(
                role="system",
                model=model,
                content=self.system_prompt,
                tokens=count_tokens(self.system_prompt, model),
            )
        ]
        logging.info(f"Initialized Gemini model: {model}")

    def parse_message(self, message: chat.Message):
        role = "model" if message["role"] == "assistant" else "user"
        return [{"role": role, "parts":[message["content"]]}]

    def chat(self, prompt):
        if prompt is None:
            pass
        else:
            if isinstance(prompt, str):
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
        # response = self.model.generate_content(prompt, stream=False)
        # return GeminiResponse(response, self.model_name)
        messages = self.parse_history()

        response = self.model.generate_content(messages, stream=False)
        response = GeminiResponse(response, self.model_name)

        self._history.append(response.to_message())

        return response

    def __repr__(self):
        return self.model_name
