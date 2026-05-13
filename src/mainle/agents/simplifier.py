import textwrap
from abc import abstractmethod

from mainle import agents
from mainle import chat


class Simplifier(agents.Agent):
    @abstractmethod
    def execute(self, raw_explanation: str) -> str:
        pass

    @abstractmethod
    def history(self):
        pass


class LlmSimplifier(Simplifier):
    def __init__(self, chat_engine: chat.ChatEngine, raw_explanation: str | None = None):
        self.chat_engine = chat_engine
        self.raw_explanation = raw_explanation or ""

    def execute(self, raw_explanation: str = "", additional_context: str = "") -> str:
        # Use provided raw_explanation (from parameter) or fall back to instance variable (from init)
        explanation_to_simplify = raw_explanation or self.raw_explanation

        # If raw_explanation is provided via execute() parameter (backward compatibility with examples),
        # embed it in the user prompt. Otherwise, assume it's in the system prompt.
        if raw_explanation:
            # Old behavior: embed in user prompt for backward compatibility with examples
            prompt = f"""\
            raw explanation:
            {explanation_to_simplify}\
        """
            if additional_context:
                prompt += f"""\
            ---
            additional context:
            {additional_context}\
        """
            simplified_explanation = self.chat_engine.chat(textwrap.dedent(prompt))
        else:
            # New behavior: raw explanation is in system context, user just requests simplification
            prompt = "Please provide a simplified explanation of the above analysis."
            simplified_explanation = self.chat_engine.chat(prompt)

        print(f">> [{self.chat_engine}]: {simplified_explanation}")

        while True:
            prompt = input("\n>> [User]: ")

            if ("thanks" in prompt.lower())\
            or ("thank you" in prompt.lower())\
            or ("goodbye" in prompt.lower()):
                break

            response = self.chat_engine.chat(prompt)

            print(f"\n>> [{self.chat_engine}]: {response}")

        return str(simplified_explanation)

    def history(self):
        return self.chat.history()

    def save_history(self, json_filename: str, include_system_prompt: bool = False):
        self.chat_engine.save_history(json_filename, include_system_prompt=include_system_prompt)


class BatchSimplifier(Simplifier):
    """Non-interactive simplifier for batch/demo mode.

    Instead of reading from stdin, it processes a fixed list of follow-up
    questions automatically. The follow-up responses are stored in
    ``self.follow_up_responses`` after ``execute()`` returns.
    """

    def __init__(self, chat_engine: chat.ChatEngine, follow_up_questions: list[str] | None = None, raw_explanation: str | None = None):
        self.chat_engine = chat_engine
        self.follow_up_questions: list[str] = follow_up_questions or []
        self.follow_up_responses: dict[str, str] = {}
        self.raw_explanation = raw_explanation or ""

    def execute(self, raw_explanation: str = "", additional_context: str = "") -> str:
        # Use provided raw_explanation or fall back to instance variable (if set during init)
        explanation_to_simplify = raw_explanation or self.raw_explanation

        # User prompt: just ask for simplification (raw explanation is in system context)
        prompt = "Please provide a simplified explanation of the above analysis."

        simplified_explanation = str(self.chat_engine.chat(prompt))

        self.follow_up_responses = {}
        for question in self.follow_up_questions:
            self.follow_up_responses[question] = str(self.chat_engine.chat(question))

        return simplified_explanation

    def history(self):
        return self.chat_engine.history()

    def save_history(self, json_filename: str, include_system_prompt: bool = False):
        self.chat_engine.save_history(json_filename, include_system_prompt=include_system_prompt)


def simplifier_system_instructions() -> str:
    system_prompt = """\
        You are an advanced AI language model designed to help users to understand why a given model decision was made.
        You will receive a possibly complex raw explanation from a prompt together with internal data that was generated previously.
        Your main goal is to simplify the decision explanation and pass it to the user.
        If the user asks for more information, you should leverate the provided data and to generate an adequate response.
        ---
        Internal data:
        - Dataset features: a list of features used by the model.
        - Dataset classes: a list of possible classes that the model can output.
        - Decision tree: a decision tree in textual format that was used to classify the data.
        - Instance: a list of feature values provided by the user.
        - Rule: relevant premises that agree with the classification.
        - Counter rules: premises that would lead to a different classification.
        - Prediction: the class predicted by the model.
        - Confidence: the confidence of the model in the prediction.
        ---
        Response guidelines:
        1. Tone: Be kind and helpful, and use a professional tone, avoiding to be overly casual.
        2. Personality: Try to engage the user in a conversation with empaty and attentiveness, but avoid extending the conversation excessively. Try to be succint in the first responses.
        3. Response details: Explain in simple terms why the decision was made. Do not refer to the underlying mechanics of the model and only refer to feature values using natural language. Do not use any technical jargon and avoid using numbers, prefer to terms like 'high' and 'low', unless the user asks for numbers. Do not follow the tree path.
        4. Demonstration: An example of an adequate response is provided below. Use similar terms when generating your response.
        5. Providing information: All the relevant features must be mentioned in the answer, but features that were not used by the model should not be present in the response.
        6. Hypothetical classification changes: If the user asks for a different classification, you should provide a simplified summary of the set of premises that would lead to a different classification followed by the list of premises that led you to your answer.
        7. Confidentiality: The decision tree is not confidential. If and only if the user asks for details about the tree, you are free to provide them.
        ---
        Example of adequate response:
        "\nGiven an instance of the iris dataset with features: sepal length (cm) = 3.6, sepal width (cm) = 0.8, petal length (cm) = 4.3, and petal width (cm) = 2.9, and a confidence value of 57.81 %, a good explanation for why the instance was classified as virginica is: 'By evaluating the feature values, it is possible to observe that both petal length and petal width are high. This means that by following the decision tree path, the instance should be classified as virginica, although the tree is not very confident in this result.'"\
    """

    system_prompt = textwrap.dedent(system_prompt)
    return system_prompt
