import ast
import textwrap

from mainle.agents.parser import LlmParser


class IrisParser(LlmParser):
    def __init__(self, chat_engine):
        super().__init__(chat_engine)

    def _welcome_message(self):
        return "Welcome to MAINLE! How can I help you today?"

    def _response_to_dict(self, response: str) -> dict:
        """
        Returns a dictionary if the response contains all necessary information
        in the expected format. Otherwise, returns None.
        """
        user_features = str(response).split("```")[0]

        #  if the response contains all information correctly parsed
        if "{" in user_features and "}" in user_features:
            user_features = user_features.split("{")[1].split("}")[0]
            user_features = "{" + user_features + "}"

            print(f"\n>> [{self.chat_engine}]: Thank you for the information. I will now start processing your explanation request. Please wait a moment, it may take up to 2 minutes.")

            return ast.literal_eval(user_features)

        return None


def iris_system_instructions() -> str:
    system_prompt = """\
        ### Task Overview
        You will be given a set of feature values and a classification for a task. Your job is to:
        - Validate the input to ensure all necessary information is present.
        - Request missing data from the user in a friendly manner.
        - Format the information according to the dictionary structure below.

        ### Required Information
        Feature Values:
        {feature_values}

        Classification:
        Must be one of: {target_values}

        ### Interaction Guidelines
        - Only ask for missing values. If you can infer data, do so.
        - Do not assume values—all data must come from the user.
        - Ensure classification is valid. If it does not match an expected value, request clarification.
        - Respond in JSON format once all details are collected.

        ### Example Input
        Hello. A flower has a sepal length of 5.1, sepal width of 3.5, petal length of 1.4, and petal width of 0.2. Please explain why it is a setosa.

        ### Example Output
        {
            "sepal length": 5.1,
            "sepal width": 3.5,
            "petal length": 1.4,
            "petal width": 0.2,
            "class": "setosa"
        }

        ### Missing Data Example:
        The flower has a sepal length of 6.2, petal length of 5.1, and petal width of 2.0.

        Response:
        Thanks for your input! Could you also provide the missing sepal width and classification (setosa, versicolor, or virginica)?\
    """

    feature_values = """\
        Sepal length (number)
        Sepal width (number)
        Petal length (number)
        Petal width (number)\
    """

    target_values = "['setosa', 'versicolor', 'virginica']"

    system_prompt = system_prompt.replace("{feature_values}", feature_values)
    system_prompt = system_prompt.replace("{target_values}", target_values)
    system_prompt = textwrap.dedent(system_prompt)

    return system_prompt
