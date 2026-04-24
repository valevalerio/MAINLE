import ast
import textwrap

from mainle.agents.parser import LlmParser


class AdultParser(LlmParser):
    def __init__(self, chat_engine):
        super().__init__(chat_engine)

    def _welcome_message(self):
        return "Welcome to MAINLE! How can I help you today?"

    def _response_to_dict(self, response: str) -> dict:
        """
        Returns a dictionary if the response contains all necessary feature values in the expected format, otherwise returns None.
        """
        user_features = str(response).split("```")[0]

        #  if the response contains all information correctly parsed
        if "{" in user_features and "}" in user_features:
            # rename 'gender' to 'sex' to match the dataset feature name
            user_features = user_features.replace("gender", "sex")
            user_features = user_features.split("{")[1].split("}")[0]
            user_features = "{" + user_features + "}"

            print(f"\n>> [{self.chat_engine}]: Thank you for the information. I will now start processing your explanation request. Please wait a moment, it may take up to 2 minutes.")

            return ast.literal_eval(user_features)

        return None


def adult_system_instructions() -> str:
    system_prompt = """\
        ### Task Overview
        You will be given a set of feature values for an individual. Your job is to:
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

        ### Example Input from another dataset
        Hello. A flower has a sepal length of 5.1, sepal width of 3.5, petal length of 1.4, and petal width of 0.2. Please explain why it is a setosa.

        ### Example Output
        {
            "age": 39,
            "workclass": "State-gov",
            "education": "Bachelors",
            "marital-status": "Never-married",
            "occupation": "Adm-clerical",
            "race": "White",
            "sex": "Male",
            "capital-gain": 2174,
            "hours-per-week": 40
        }

        ### Missing Data Example:
        The individual is 45 years old, works in the private sector, and works 50 hours per week.

        Response:
        Thanks for your input! Could you also provide the missing education, marital-status, occupation, race, sex, and capital-gain?\
    """

    feature_values = """\
        - age: number
        - workclass: [Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay]
        - education: [Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool]
        - marital-status: [Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse]
        - occupation: [Adm-clerical, Exec-managerial, Handlers-cleaners, Prof-specialty, Other-service, Sales, Transport-moving, Farming-fishing, Machine-op-inspct, Tech-support, Craft-repair, Protective-serv, Armed-Forces, Priv-house-serv]
        - race: [Black, White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other]
        - sex: [Female, Male]
        - capital-gain: number
        - hours-per-week: number
    """

    system_prompt = system_prompt.replace("{feature_values}", feature_values)
    system_prompt = textwrap.dedent(system_prompt)

    return system_prompt
