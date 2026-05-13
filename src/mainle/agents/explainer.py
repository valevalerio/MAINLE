from abc import abstractmethod

import pandas as pd
from sklearn import tree

from lore_sa.lore import TabularRandomGeneratorLore
from mainle import agents


class Explainer(agents.Agent):
    def __init__(self):
        pass

    @abstractmethod
    def execute(self, parsed_input: dict):
        pass

    def history(self):
        pass


class LoreExplainer(Explainer):
    def __init__(self, dataset, model):
        self.dataset = dataset
        self.model = model

    def execute(self, instance: dict):
        feature_names = self.dataset.get_features_names()
        feature_names = [feature for feature in feature_names if feature != 'class']
        feature_values = pd.Series({feature: instance[feature] for feature in feature_names})

        # create a LORE explainer using a random neighborhood generator
        explainer = TabularRandomGeneratorLore(self.model, self.dataset)

        explanation = explainer.explain(feature_values)

        predicted_class = self.model.predict(feature_values.values.reshape(1, -1))[0]

        predicted_proba = max(self.model.predict_proba(feature_values.values.reshape(1, -1))[0])

        rule = explanation["rule"]

        # restrict the number of counterfactuals
        counter_rules = explanation["counterfactuals"]
        max_num_counterfactuals = min(5, len(counter_rules))
        counter_rules = counter_rules[0:max_num_counterfactuals]

        class_values = sorted(list(set(self.dataset.get_class_values())))

        context, question = self._generate_prompt(
            feature_names,
            class_values,
            tree.export_text(
                explainer.surrogate.dt,
                feature_names=list(explainer.encoder.get_encoded_features().values())
            ),
            feature_values,
            rule,
            counter_rules,
            predicted_class,
            predicted_proba,
            add_demonstration=False,
            add_instructions=False,
        )

        return context, question

    def _generate_prompt(
        self,
        feature_names,
        class_values,
        tree_representation,
        instance_values,
        rule,
        counter_rules,
        predicted_class,
        predicted_proba,
        add_demonstration=False,
        add_instructions=False,
    ):
        context = f"""{self._dataset_description(feature_names, class_values)} A decision tree was trained on the dataset and the tree is:
    {tree_representation}{self._default_demonstration() if add_demonstration else ""}
    {self._instance_description(feature_names, instance_values)}{self._rule_and_counterrules(rule, counter_rules)}{self._instructions() if add_instructions else ""}"""

        question = self._question(predicted_class, confidence=predicted_proba, has_example=add_demonstration)

        return context, question

    def _dataset_description(self, feature_names, target_names):
        # convert all list elements to string
        feature_names = [str(feature) for feature in feature_names]
        target_names = [str(target) for target in target_names]

        features_text = ", ".join(feature_names[:-1]) + f", and {feature_names[-1]}"
        targets_text = ", ".join(target_names[:-1]) + f", and {target_names[-1]}"

        description = f"Consider a dataset that has the following features: {features_text}, where E1 means \'at moment of admission\'. Each instance can be classified into one of the following classes: {targets_text}."

        return description

    def _default_demonstration(self):
        demonstration = "\nGiven an instance of the iris dataset with features: sepal length (cm) = 3.6, sepal width (cm) = 0.8, petal length (cm) = 4.3, and petal width (cm) = 2.9, and a confidence value of 57.81 %, a good explanation for why the instance was classified as virginica is: 'By evaluating the feature values, it is possible to observe that both petal length and petal width are high. This means that by following the decision tree path, the instance should be classified as virginica, although the tree is not very confident in this result.'"

        return demonstration

    def _instance_description(self, feature_names, instance_values):
        values = list(instance_values.values) if hasattr(instance_values, "values") else list(instance_values)
        instance_text = ""

        for feature, instance in zip(feature_names[:-1], values[:-1]):
            instance_text += f"{feature} = {instance}, "

        instance_text += f"and {feature_names[-1]} = {values[-1]}"

        description = f"An instance has features: {instance_text}."

        return description

    def _rule_and_counterrules(self, rule, counter_rules):
        text = f"\n\nThe relevant premises for the decision rule are:\n{rule}."
        text += f"\n\nThe classification would be different if any of the following set of counter rule premises was true:\n"

        max_num_counterfactuals = min(5, len(counter_rules))

        for i in range(max_num_counterfactuals):
            simpler_rule = counter_rules[i]
            simpler_rule["premises"] = [
                {
                    "attr": premise["attr"],
                    "val": round(premise["val"], 4) if isinstance(premise["val"], (int, float)) else premise["val"],
                    "op": premise["op"],
                }
                for premise in simpler_rule["premises"]
            ]
            text += f"{counter_rules[i]}\n"

        return text

    def _instructions(self):
        instructions_text = "\nTo answer the following question, do not refer to the underlying mechanics of the decision tree in any way. Refer to the features using natural language. All the relevant features must be mentioned in the answer, ignore the features not used in the tree. Finnaly, avoid technical jargon or numerical values in the response and prefer to use terms like 'high' and 'low'."

        return instructions_text

    def _question(self, target_class, confidence=None, has_example=False):
        confidence_text = f" with a confidence of {100.0 * confidence:.2f} %." if confidence else "."

        question_text = f"Please explain in {'similar' if has_example else 'simple'} terms why the classifier concluded that the given example is '{target_class}'{confidence_text}"

        return question_text
