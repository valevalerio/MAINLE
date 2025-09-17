import json
import textwrap

from mainle.chat.engines import Gpt, Gemini, Llama, DeepSeek


def winner_decision_system_prompt(raw_explanation, simplified_explanation):
    system_prompt = """\
        You are an advanced AI evaluator tasked with comparing two explanations provided by a model: `raw_explainer` and `interactive_explainer`.
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
        ### Instructions:
        1. Evaluate both explanations based on the four criteria above.
        2. For **each criterion**, identify the winner, ensuring no ties.
        3. Restrict your response to the format provided below, replacing 'X' with the winner name, and DO NOT ADD ANYTHING ELSE.
        ---
        ### Format for Response:
        - **Technical Jargon**: [Winner: Explanation 'X' | Improvements]
        - **Simplicity**: [Winner: Explanation 'X' | Improvements]
        - **Completeness**: [Winner: Explanation 'X' | Improvements]
        - **Conciseness**: [Winner: Explanation 'X' | Improvements]
        ---
        ### Explanations to Compare:
        `raw_explanation`:
        {raw_explanation}

        `simplified_explanation`:
        {simplified_explanation}
    """.strip()

    system_prompt = textwrap.dedent(system_prompt)
    system_prompt = system_prompt.replace("{raw_explanation}", raw_explanation)
    system_prompt = system_prompt.replace("{simplified_explanation}", simplified_explanation)

    return system_prompt


def main(folders, file_idxs, model):
    for folder in folders:
        for idx in file_idxs:
            file_prefix = f"../history/{folder}/{idx}_"
            json_file = f"{file_prefix}explanations.json"

            with open(json_file) as f:
                data = json.load(f)

                raw_explanation = data['raw_explanation']
                simplified_explanation = data['interactive_explanation']

                system_prompt = winner_decision_system_prompt(raw_explanation, simplified_explanation)

                if model == "llama":
                    chat_engine = Llama(model="llama3.2")
                    response = chat_engine.chat(system_prompt)
                elif model == "gpt":
                    chat_engine = Gpt(model="gpt-4o-mini", system_prompt=system_prompt)
                    response = chat_engine.chat(None)
                elif model == "gemini":
                    chat_engine = Gemini(model="gemini-2.0-flash", system_prompt=system_prompt)
                    response = chat_engine.chat(None)
                elif model == "deepseek":
                    chat_engine = DeepSeek(model="deepseek-r1")
                    response = chat_engine.chat(system_prompt)

                print(f"[assistant]: {response}")

                filename = f"{file_prefix}{model}_winner.json"
                print(f"Saving to {filename}")

                with open(filename, "w", encoding="utf-8") as f:
                    json.dump(chat_engine.history(include_system_prompt=True), f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    for model in ["gpt", "gemini", "llama", "deepseek"]:
        main(["breastcancer", "credit", "iris"], range(0,5), model)
        main(["experiments"], [4,5,6,7,8,9,10,12,13,14], model)
