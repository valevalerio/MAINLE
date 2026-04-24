import ast
import textwrap

from mainle.agents.parser import LlmParser


class WineParser(LlmParser):
    def __init__(self, chat_engine):
        super().__init__(chat_engine)

    def _welcome_message(self):
        return "Welcome to MAINLE! How can I help you today?"

    def _response_to_dict(self, response: str) -> dict:
        """
        Returns a dictionary if the response contains all necessary feature values
        in the expected format. Otherwise, returns None.
        """
        user_features = str(response).split("```")[0]

        #  if the response contains all information correctly parsed
        if "{" in user_features and "}" in user_features:
            user_features = user_features.replace("class ", "class_")
            user_features = user_features.split("{")[1].split("}")[0]
            user_features = "{" + user_features + "}"

            print(f"\n>> [{self.chat_engine}]: Thank you for the information. I will now start processing your explanation request. Please wait a moment, it may take up to 2 minutes.")

            return ast.literal_eval(user_features)

        return None


def wine_system_instructions() -> str:
    system_prompt = """\
        ### Task Overview
        You will be given a set of feature values for a wine sample. Your job is to:
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
        The wine has alcohol 13.2, malic acid 1.8, ash 2.4, alcalinity of ash 18.5, magnesium 105, total phenols 2.6, flavanoids 2.8, nonflavanoid phenols 0.3, proanthocyanins 1.9, color intensity 5.2, hue 1.05, OD280/OD315 of diluted wines 3.1, and proline 980.

        ### Example Output
        {
            "Alcohol": 13.2,
            "Malic acid": 1.8,
            "Ash": 2.4,
            "Alcalinity of ash": 18.5,
            "Magnesium": 105,
            "Total phenols": 2.6,
            "Flavanoids": 2.8,
            "Nonflavanoid phenols": 0.3,
            "Proanthocyanins": 1.9,
            "Color intensity": 5.2,
            "Hue": 1.05,
            "OD280/OD315 of diluted wines": 3.1,
            "Proline": 980
        }

        ### Missing Data Example:
        The wine has alcohol 12.8, ash 2.2, and hue 0.95.

        Response:
        Thanks for your input! Could you also provide the remaining missing feature values?\
    """

    feature_values = """\
        - Alcohol: number
        - Malic acid: number
        - Ash: number
        - Alcalinity of ash: number
        - Magnesium: number
        - Total phenols: number
        - Flavanoids: number
        - Nonflavanoid phenols: number
        - Proanthocyanins: number
        - Color intensity: number
        - Hue: number
        - OD280/OD315 of diluted wines: number
        - Proline: number
    """

    system_prompt = system_prompt.replace("{feature_values}", feature_values)
    system_prompt = textwrap.dedent(system_prompt)

    return system_prompt
