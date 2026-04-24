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
        Returns a dictionary if the response contains all necessary feature values.
        The class is not requested from the user and will be predicted internally.
        """
        parsed_features = self._response_to_exact_schema_dict(
            response,
            [
                "sepal length (cm)",
                "sepal width (cm)",
                "petal length (cm)",
                "petal width (cm)",
            ],
            aliases={
                "sepal length (cm)": ["sepal length"],
                "sepal width (cm)": ["sepal width"],
                "petal length (cm)": ["petal length"],
                "petal width (cm)": ["petal width"],
            },
        )

        if parsed_features:
            print(f"\n>> [{self.chat_engine}]: Thank you for the information. I will now start processing your explanation request. Please wait a moment, it may take up to 2 minutes.")

        return parsed_features


def iris_system_instructions() -> str:
    system_prompt = """\
        ### Task Overview
        You will be given a set of feature values for a flower. Your job is to:
        - Validate the input to ensure all necessary feature values are present.
        - Request missing data from the user in a friendly manner.
        - Format the information according to the dictionary structure below.

        ### Required Information
        Feature Values:
        {feature_values}

        ### Interaction Guidelines
        - Only ask for missing values. If you can infer data, do so.
        - Do not assume values—all data must come from the user.
        - Respond in JSON format once all details are collected.

        ### Example Input
        Hello. A flower has a sepal length of 5.1, sepal width of 3.5, petal length of 1.4, and petal width of 0.2. Please explain why it is a setosa.

        ### Example Output
        {
            "sepal length": 5.1,
            "sepal width": 3.5,
            "petal length": 1.4,
            "petal width": 0.2
        }

        ### Missing Data Example:
        The flower has a sepal length (cm) of 6.2 and petal length (cm) of 5.1.

        Response:
        Thanks for your input! Could you also provide the missing sepal width (cm) and petal width (cm)?\
    """

    feature_values = """\
        sepal length (cm): number
        sepal width (cm): number
        petal length (cm): number
        petal width (cm): number\
    """

    system_prompt = system_prompt.replace("{feature_values}", feature_values)
    system_prompt = textwrap.dedent(system_prompt)

    return system_prompt

