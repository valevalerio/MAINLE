import pandas as pd
from sklearn import datasets

from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

from lore_sa.dataset import TabularDataset
from lore_sa.bbox import sklearn_classifier_bbox

from mainle.agents.explainer import LoreExplainer


class WineExplainer(LoreExplainer):
    def __init__(self):
        data = datasets.load_wine()
        self.dataset = pd.DataFrame(data=data.data, columns=data.feature_names)
        self.dataset.dropna(inplace = True)
        self.dataset["class"] = pd.Series(data.target).astype(str)
        self.dataset["class"] = self.dataset["class"].replace({"0": "class_0", "1": "class_1", "2": "class_2"})
        self.dataset = TabularDataset(self.dataset, class_name="class")

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), [0,1,2,3,4,5,6,7,8,9,10,11,12]),
            ]
        )

        classifier = make_pipeline(preprocessor, MLPClassifier(max_iter=100, random_state=42))
        self.model = self._train_model(classifier, self.dataset)

        super().__init__(self.dataset, self.model)

    def _train_model(self, classifier, dataset):
        X_train, X_test, y_train, y_test = train_test_split(
            dataset.df.iloc[:, :-1].values,
            dataset.df["class"].values,
            test_size=0.3,
            random_state=42,
            stratify=dataset.df["class"].values
        )

        classifier.fit(X_train, y_train)

        return sklearn_classifier_bbox.sklearnBBox(classifier)
