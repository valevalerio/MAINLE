from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier

from lore_sa.dataset import TabularDataset
from lore_sa.bbox import sklearn_classifier_bbox

from mainle.agents.explainer import LoreExplainer


class AdultExplainer(LoreExplainer):
    def __init__(self):
        self.dataset = TabularDataset.from_csv("tabular_datasets/adult.csv", class_name = "class")
        self.dataset.df.dropna(inplace = True)
        self.dataset.df.drop(["fnlwgt", "education-num", "relationship", "capital-loss", "native-country"], inplace=True, axis=1)
        self.dataset.update_descriptor()

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), [0,7,8]),
                ("cat", OrdinalEncoder(), [1,2,3,4,5,6])
            ]
        )

        classifier = make_pipeline(preprocessor, RandomForestClassifier(n_estimators=100, random_state=42))
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
