import json
import textwrap

from mainle.chat.engines import Gpt, Gemini, Llama, DeepSeek


def rating_scale_system_prompt(raw_explanation, simplified_explanation):
    system_prompt = """
        You are an evaluator that has no knowledge of Machine Learning tasked with comparing two explanations provided by a classification model: `raw_explanation` and `simplified_explanation`.
        Your main objective is to rate each explanation according to four criteria.
        ---
        ### Evaluation Criteria:
        **Technical Jargon**: you were able to follow the explanation easily and did not need previous knowledge of the underlying mechanics.
        **Simplicity**: the terms used were simple and easy to read.
        **Completeness**: you feel you understood the reasons why the decision was made and did not miss additional information.
        **Conciseness**: you feel that all information presented was necessary and there was no useless information in the explanation.
        ---
        ### Important Context:
        - `raw_explanation`: A direct explanation provided without interaction, based purely on the model's output.
        - `simplified_explanation`: An explanation generated through a conversation between an agent and a user. The conversation builds context that may enhance interpretability.
        ---
        ### Instructions:
        1. Evaluate both explanations based on the criteria above considering that you are not familiar with Machine Learning, and you are looking for an explanation that is easy to understand.
        2. For **each criterion**, give a rate from 1 to 5, in which:
            - 1 means 'strongly disagree'
            - 2 means 'disagree'
            - 3 means 'neutral'
            - 4 means 'agree'
            - 5 means 'strongly agree'
        3. For the response format, 'raw_rating' and 'simplified_rating' should be the numerical ratings you gave for each explanation.
        4. Restrict your response to the format provided below, replacing 'raw_rating' and 'simplified_rating' with the numerical ratings you gave, and DO NOT ADD ANYTHING ELSE.
        ---
        ### Format for Response:
        - **Technical Jargon**: [raw_rating , simplified_rating]
        - **Simplicity**: [raw_rating , simplified_rating]
        - **Completeness**: [raw_rating , simplified_rating]
        - **Conciseness**: [raw_rating , simplified_rating]
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

                system_prompt = rating_scale_system_prompt(raw_explanation, simplified_explanation)

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

                filename = f"{file_prefix}{model}_human_rating.json"
                print(f"Saving to {filename}")

                with open(filename, "w", encoding="utf-8") as f:
                    json.dump(chat_engine.history(include_system_prompt=True), f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    for model in ["gpt", "gemini", "llama", "deepseek"]:
        main(["breastcancer", "credit", "iris"], range(0,5), model)
        main(["experiments"], [4,5,6,7,8,9,10,12,13,14], model)
