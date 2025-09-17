from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.ensemble import HistGradientBoostingClassifier

from lore_sa.dataset import TabularDataset
from lore_sa.bbox import sklearn_classifier_bbox

from mainle.agents.explainer import LoreExplainer


class CreditExplainer(LoreExplainer):
    def __init__(self):
        self.dataset = TabularDataset.from_csv("tabular_datasets/crxdata.csv", class_name = "class")
        self.dataset.df.dropna(inplace = True)
        self.dataset.df["class"] = self.dataset.df["class"].replace({"+": "approved", "-": "rejected"})
        self.dataset.df = self.dataset.df.set_axis(["Gender", "Age", "Debt", "Marital status", "Bank customer", "Educational level", "Ethnicity", "Number of years employed", "Prior default", "Employment status", "Credit score", "Driver license", "Citizenship", "Zipcode", "Income", "class"], axis="columns")
        self.dataset.update_descriptor()

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), [1,2,7,10,13,14]),
                ('cat', OrdinalEncoder(), [0,3,4,5,6,8,9,11,12])
            ]
        )

        classifier = make_pipeline(preprocessor, HistGradientBoostingClassifier(max_leaf_nodes=15, random_state=0))
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
