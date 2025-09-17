import textwrap
from abc import abstractmethod

from mainle import agents
from mainle import chat


class Critic(agents.Agent):
    @abstractmethod
    def execute(self, raw_explanation: str, simplified_explanation: str) -> str:
        pass

    @abstractmethod
    def history(self):
        pass


class LlmCritic(Critic):
    def __init__(self, chat_engine: chat.ChatEngine):
        self.chat_engine = chat_engine

    def execute(self, raw_explanation: str, simplified_explanation: str) -> str:
        prompt = f"""\
            raw explanation:
            {raw_explanation}\
            ---
            simplified explanation:
            {simplified_explanation}\
        """
        return self.chat_engine.chat(textwrap.dedent(prompt))

    def history(self):
        return self.chat.history()

    def save_history(self, json_filename: str, include_system_prompt: bool = False):
        self.chat_engine.save_history(json_filename, include_system_prompt=include_system_prompt)


def critic_system_instructions() -> str:
    system_prompt = """\
        You are an advanced AI evaluator tasked with comparing two explanations provided by a model: `raw_explanation` and `simplified_explanation`.
        Your main objective is to identify which explanation better satisfies the following criteria. **Ties are not allowed; always choose a winner.**
        If neither explanation performs well, select the one that is relatively better and suggest improvements for both.
        ### Evaluation Criteria:
        1. **Technical Jargon**:
            - The explanation must avoid referring to the decision tree path, technical structures, or the mechanics of the classifier.
            - The language should describe the reasoning in a way that a non-technical user can follow without needing knowledge of the decision tree.
            - Avoid numbers, equations, or technical terminology, like "instance", "feature", and "algorithm".
            - Feature and class names should **not** be considered technical jargon.
        2. **Simplicity**:
            - Explanations should use the simplest terms possible without losing accuracy.
            - Avoid overly detailed or verbose sentences that might confuse users.
            - The explanation must use simple and natural language, replacing technical terms with familiar and intuitive concepts. For example, instead of saying "greater than" say "a high value".
        3. **Completeness**:
            - The explanation should provide enough information to understand the decision-making process.
            - It should include all relevant features and their values that influenced the decision.
            - An explanation that does not include all necessary information or includes more information than necessary should be penalized.
        4. **Conciseness**:
            - Focus only on the relevant features that influence the decision.
            - An explanation that includes irrelevant features or unnecessary details is penalized.
            - An explanation that does not include relevant information should also be penalized.
            - The winner should provide the most relevant information without sacrificing clarity.
        ---
        ### Important Context:
        - `raw_explanation`: A direct explanation provided without interaction, based purely on the model's output.
        - `simplified_explanation`: An explanation generated through a conversation between an agent and a user. The conversation builds context that may enhance interpretability.
        ---
        ### Instructions:
        1. Evaluate both explanations based on the criteria above.
        2. For **each criterion**, identify the winner, ensuring no ties, and explain your reasoning.
        3. Provide a **final recommendation** on which explanation is better overall.
        4. Suggest specific improvements to make the explanations clearer and more user-friendly.
        ---
        ### Format for Response:
        - **Technical Jargon**: [Winner: Explanation X | Improvements]
        - **Simplicity**: [Winner: Explanation X | Improvements]
        - **Completeness**: [Winner: Explanation X | Improvements]
        - **Conciseness**: [Winner: Explanation X | Improvements]
        - **Final Recommendation**: Explanation X
        - **Justification**: Provide reasons for your final choice, highlighting strengths and suggesting improvements for both explanations.\
    """.strip()

    system_prompt = textwrap.dedent(system_prompt)
    return system_prompt
