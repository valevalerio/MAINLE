import os

from mainle.utils.concat_history import concat_json_files

from mainle.parsers.credit import CreditParser, credit_system_instructions
from mainle.explainers.credit import CreditExplainer
from mainle.agents.translator import LlmTranslator
from mainle.agents.simplifier import LlmSimplifier, simplifier_system_instructions
from mainle.agents.critic import LlmCritic, critic_system_instructions
from mainle.chat.engines.gpt import Gpt


def main(save_history: bool = True, run_critic_agent: bool = True):
    mainleParser = CreditParser(Gpt(model="gpt-4o-mini", system_prompt=credit_system_instructions()))
    user_info = mainleParser.execute()
    if save_history:
        mainleParser.save_history("1_parser.json", include_system_prompt=True)

    mainleExplainer = CreditExplainer()
    translator_prompt = mainleExplainer.execute(user_info)

    mainleTranslator = LlmTranslator(Gpt(model="gpt-4o-mini"))
    raw_explanation = mainleTranslator.execute(translator_prompt)
    if save_history:
        mainleTranslator.save_history("2_translator.json", include_system_prompt=True)

    mainleSimplifier = LlmSimplifier(Gpt(model="gpt-4o-mini", system_prompt=simplifier_system_instructions()))
    simplified_explanation = mainleSimplifier.execute(raw_explanation, additional_context=translator_prompt)
    if save_history:
        mainleSimplifier.save_history("3_simplifier.json", include_system_prompt=True)

    if run_critic_agent:
        mainleCritic = LlmCritic(Gpt(model="gpt-4o-mini", system_prompt=critic_system_instructions()))
        mainleCritic.execute(raw_explanation, simplified_explanation)
        if save_history:
            mainleCritic.save_history("4_critic.json", include_system_prompt=True)

    if save_history:
        # check if the folder "history" exists, if not create it
        if not os.path.exists("../history"):
            os.makedirs("../history")

        json_files = [
            "1_parser.json",
            "2_translator.json",
            "3_simplifier.json",
        ]
        if run_critic_agent:
            json_files.append("4_critic.json")

        concat_json_files(json_files, "../history/full_history.json")

        # remove individual json files
        [os.remove(f) for f in json_files]
