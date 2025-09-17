# Explanation Comparison

Two different systems generated two different explanations of why a Machine Learning model made a decision. You will see the question followed by the two explanations. Then, we kindly ask you to rate each explanation according to four criteria, as detailed below.

## Question

Hello, a flower has sepal length of 7.3, sepal width of 2.9, petal length of 6.3, and petal width of 1.8. Could you please explain why it is virginica?

## First Explanation

The classifier used a decision tree to make its prediction about the flower species based on the specific features provided (sepal length, sepal width, petal length, and petal width). Let’s break down the reasoning behind the classification into ‘virginica’ and why it has a confidence of 81.78%.
1. **Input Features**: The flower instance being classified has the following measurements:
    - **Sepal Length**: 7.3 cm
    - **Sepal Width**: 2.9 cm
    - **Petal Length**: 6.3 cm
    - **Petal Width**: 1.8 cm
2. **Decision Tree Path**: The decision tree evaluates the instance by traversing it based on the input features. Let’s see how the classifier reaches the conclusion step by step, following the tree structure:
    - The first condition in the tree checks if the **sepal width (2.9 cm)** is **<= 3.05 cm**. This condition is **true**, so we move down that branch.
    - Next, we evaluate the **sepal length (7.3 cm)** against the condition **<= 6.59 cm**. This condition is **false** (since 7.3 > 6.59), so we take the branch where the **sepal length** is **> 6.59**.
    - The next criterion is the **petal width (1.8 cm)** which needs to be **<= 1.56 cm**. This condition is **false** (since 1.8 > 1.56), so we proceed along the branch that leads to where the **petal width** is **> 1.56**.
    - Now, we check **sepal width <= 2.3 cm**. This condition is also **false** (since 2.9 > 2.3), which leads us to the branch that says if **sepal width** is **> 2.3**, we will ultimately predict **class = virginica**.
3. **Conclusion**: After following through these steps, we reach a leaf in the decision tree that predicts ‘virginica’.
4. **Confidence Level**: The classifier indicates a confidence of 81.78%. This means that based on the training data related to similar measurements and the patterns it has learned, the decision tree is 81.78% sure that this instance fits the class ‘virginica’. The confidence percentage is derived from how frequently instances with similar features belong to the ‘virginica’ class in the training dataset.
5. **Counter Rule**: The counter-rule indicates what would happen if the conditions were true differently, which gives us insight into the specificity of the model. If any of those different conditions were met, the prediction could potentially change—indicating that there are boundary conditions that could alter the classification outcome.

In summary, the classifier determined that the example is ‘virginica’ based on a series of checked conditions resulting in a decision pathway through the decision tree, and the confidence reflects how often similar instances in the training data were correctly classified as ‘virginica’.

### Criteria Description

Use the following criteria to rate First Explanation:
- **Technical Jargon**: you were able to follow the explanation easily and did not need previous knowledge of the internal procedures.
- **Simplicity**: the terms used were simple and easy to read.
- **Completeness**: you feel you understood the reasons why the decision was made and did not miss additional information.
- **Conciseness**: you feel that all information presented was necessary, there was no useless information in the explanation, and the explanation could not be shorter.

### Please rate the first explanation according to the four evaluation criteria.

| | Strongly Disagree | Disagree | Neutral | Agree | Strongly Agree |
| --- | --- | --- | --- | --- | --- |
| Technical Jargon | | | | | |
| Simplicity | | | | | |
| Completeness | | | | | |
| Conciseness | | | | | |

## Second Explanation

The classifier determined that the flower instance is classified as ‘virginica’ based on the following observations:

- The **sepal width** is slightly high, and it falls into a range that typically indicates this species.
- The **sepal length** is also high, which is another strong indicator for ‘virginica’.
- The **petal width** exceeds a certain threshold, providing additional support for this classification.
- Finally, the combination of these features led the model to confidently predict ‘virginica’.

The model has a confidence level of around 81.78%, meaning it is fairly certain that this flower belongs to the ‘virginica’ class based on the similarity of its features to those seen in other ‘virginica’ instances in the training data.

If the conditions had been slightly different—like if the sepal width had been lower or the petal width lower as well—the prediction might have changed.

### Criteria Description

Use the following criteria to rate Second Explanation:
- **Technical Jargon**: you were able to follow the explanation easily and did not need previous knowledge of the internal procedures.
- **Simplicity**: the terms used were simple and easy to read.
- **Completeness**: you feel you understood the reasons why the decision was made and did not miss additional information.
- **Conciseness**: you feel that all information presented was necessary, there was no useless information in the explanation, and the explanation could not be shorter.

### Please rate the second explanation according to the four evaluation criteria.

| | Strongly Disagree | Disagree | Neutral | Agree | Strongly Agree |
| --- | --- | --- | --- | --- | --- |
| Technical Jargon | | | | | |
| Simplicity | | | | | |
| Completeness | | | | | |
| Conciseness | | | | | |
