import ast
import textwrap

from mainle.agents.parser import LlmParser


class BreastcancerParser(LlmParser):
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
            user_features = user_features.split("{")[1].split("}")[0]
            user_features = "{" + user_features + "}"

            print(f"\n>> [{self.chat_engine}]: Thank you for the information. I will now start processing your explanation request. Please wait a moment, it may take up to 2 minutes.")

            return ast.literal_eval(user_features)

        return None


def breastcancer_system_instructions() -> str:
    system_prompt = """\
        ### Task Overview
        You will be given a set of feature values for a breast cancer case. Your job is to:
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
        The case has clump thickness 5, uniformity of cell size 1, uniformity of cell shape 1, marginal adhesion 1, single epithelial cell size 2, bare nuclei 1, bland chromatin 3, normal nucleoli 1, and mitoses 1.

        ### Example Output
        {
            "Clump_thickness": 5,
            "Uniformity_of_cell_size": 1,
            "Uniformity_of_cell_shape": 1,
            "Marginal_adhesion": 1,
            "Single_epithelial_cell_size": 2,
            "Bare_nuclei": 1,
            "Bland_chromatin": 3,
            "Normal_nucleoli": 1,
            "Mitoses": 1
        }

        ### Missing Data Example:
        The case has clump thickness 7, uniformity of cell size 8, and mitoses 2.

        Response:
        Thanks for your input! Could you also provide the missing feature values?\
    """

    feature_values = """\
        - Clump_thickness: number
        - Uniformity_of_cell_size: number
        - Uniformity_of_cell_shape: number
        - Marginal_adhesion: number
        - Single_epithelial_cell_size: number
        - Bare_nuclei: number
        - Bland_chromatin: number
        - Normal_nucleoli: number
        - Mitoses: number
    """

    system_prompt = system_prompt.replace("{feature_values}", feature_values)
    system_prompt = textwrap.dedent(system_prompt)

    return system_prompt
