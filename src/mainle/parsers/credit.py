import ast
import textwrap

from mainle.agents.parser import LlmParser


class CreditParser(LlmParser):
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


def credit_system_instructions() -> str:
    system_prompt = """\
        ### Task Overview
        You will be given a set of feature values for a credit application. Your job is to:
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
        The applicant has gender b, age 31, debt 1.25, marital status u, bank customer g, educational level q, ethnicity v, 2 years employed, prior default f, employment status t, credit score 20, driver license t, citizenship g, zipcode 120, and income 5000.

        ### Example Output
        {
            "Gender": "b",
            "Age": 31,
            "Debt": 1.25,
            "Marital status": "u",
            "Bank customer": "g",
            "Educational level": "q",
            "Ethnicity": "v",
            "Number of years employed": 2,
            "Prior default": "f",
            "Employment status": "t",
            "Credit score": 20,
            "Driver license": "t",
            "Citizenship": "g",
            "Zipcode": 120,
            "Income": 5000
        }

        ### Missing Data Example:
        The applicant has age 40, debt 0.5, and income 2000.

        Response:
        Thanks for your input! Could you also provide the remaining missing feature values?\
    """

    feature_values = """\
        - Gender: ['a' | 'b']
        - Age: number
        - Debt: number
        - Marital status: ['u' | 'y' | 'l']
        - Bank customer: ['g' | 'p' | 'gg']
        - Educational level: ['w' | 'q' | 'm' | 'r' | 'cc' | 'k' | 'c' | 'd' | 'x' | 'i' | 'e' | 'aa' | 'ff' | 'j']
        - Ethnicity: ['v' | 'h' | 'bb' | 'ff' | 'j' | 'z' | 'o' | 'dd' | 'n']
        - Number of years employed: number
        - Prior default: ['t' | 'f']
        - Employment status: ['t' | 'f']
        - Credit score: number
        - Driver license: ['f' | 't']
        - Citizenship: ['g' | 's' | 'p']
        - Zipcode: number
        - Income: number
    """

    system_prompt = system_prompt.replace("{feature_values}", feature_values)
    system_prompt = textwrap.dedent(system_prompt)

    return system_prompt
