# MAINLE 
Pronunciation: _/ˈmeɪnli/_

MAINLE is a *multi-agent* architecture for a *conversational interface* to provide *simplified* explanations for *non-expert* users.

---

This repository contains the source code of MAINLE. This code has been used to run the experiments presented in the paper
```
MAINLE: a Multi-Agent, Interactive, Natural Language Local Explainer of Classification Tasks
```
written by [Paulo Bruno Serafim](https://paulobruno.github.io), [Rômulo Férrer Filho](https://github.com/romulofff), [Stenio Wagner Freitas](https://github.com/steniowagner), [Gizem Gezici](https://scholar.google.com/citations?user=JQK0dyUAAAAJ), [Fosca Giannotti](https://kdd.isti.cnr.it/people/giannotti-fosca), [Franco Raimondi](https://www2.rmnd.net/), and [Alexandre Santos](https://github.com/magnomont12), and presented at the [ECML PKDD 2025](https://ecmlpkdd.org/2025/) conference.


## Pre-print

A pre-print version of the paper can be found [here](https://ecmlpkdd-storage.s3.eu-central-1.amazonaws.com/preprints/2025/research/preprint_ecml_pkdd_2025_research_1246.pdf).

# How to use

## Getting the code

Download or clone [MAINLE's repository](github.com/paulobruno/ecml-pkdd-2025).

## Running

The easiest way to test MAINLE is to run the `main.py` file from the `src` folder.
It will run an example using the Iris dataset.
First, please make sure all dependencies mentioned below are correctly installed.

## Provided Examples

We also provide five ready-to-use examples using different datasets and classifiers.
They can be found in the `src/mainle/examples` folder.
If you want to create your own custom code, they are a great source for a starting point.

If you just want to test other cases rather than the default Iris, you can run an example. To do it, please edit the first line of the `src/main.py` file to use the dataset you would like to test.
The available datasets are:
- Iris flower
    - `from mainle.examples.iris import main`
- Wine classification
    - `from mainle.examples.wine import main`
- Adult income
    - `from mainle.examples.adult import main`
- Loan approval
    - `from mainle.examples.credit import main`
- Breast cancer prediction
    - `from mainle.examples.breastcancer import main`

## Dependencies and working versions

You can find full requirements in the `requirements.txt` file.

### Libraries
- pandas: 2.3.2
- scikit-learn: 1.7.2
- deap: 1.4.3
- tiktoken: 0.11.0

### LLMs
- ollama: 0.5.4
- openai: 1.107.3
- google-genai: 1.38.0

### LORE dependency
MAINLE relies on **LORE (Local Rule-Based Explanations)** for the Explainer agent.
The repository includes LORE as a Git submodule, already configured and pinned to a compatible version.
#### Setup

After cloning the repository, initialize the submodule:

```
git submodule update --init --recursive
```

Then install it as a local package:

```
pip install -e src/lore_sa/
```

#### Notes
The submodule is pinned to a tested commit for compatibility. Do not move the `lore_sa` folder, as MAINLE expects it at `src/lore_sa/`. Next work will include updating the submodule to the latest version of the pip package, ensuring compatibility and removing it from the repository itself. 

### API Keys

Depending on the LLM you use, it might be necessary to provide an API key for the respective model.
Before running MAINLE, make sure that you can chat with the LLM externally.
Optionally, if the chat engine allows, you can pass the key by using the argument `api_key` in the construction of each LLM Agent.

### Ending a conversation

To end a conversation, please reply with `Thank you`, `Thanks`, or `Goodbye`.

### Visualizing the conversation

By default, MAINLE will save a conversation JSON inside a `history` folder.
We provide an HTML visualizer in the `sample_conversations` folder and also a web-based visualizer, accessible at [https://pb-rho.vercel.app/](https://pb-rho.vercel.app/).

## Sample questions

If you try running MAINLE, you will be expected to present an input instance and a classification decision.
If you do not have your own instance, you can use one of the samples below.

*Note: Please remember to update the dataset that you will be using in the first line of the `main.py` file.*

- Wine:
```
A wine has features: alcohol = 13.16, malic_acid = 2.36, ash = 2.67, alcalinity_of_ash = 18.6, magnesium = 101.0, total_phenols = 2.8, flavanoids = 3.24, nonflavanoid_phenols = 0.3, proanthocyanins = 2.81, color_intensity = 5.68, hue = 1.03, od280/od315_of_diluted_wines = 3.17, and proline = 1185.0. Please explain why it was concluded to be of class 0.
```

- Brest cancer:
```
Hi, why a collected sample that has clump thickness 10, uniformity of cell size 10, uniformity of cell shape 10, marginal adhesion 4, single epithelial cell size 8, bare nuclei 1, bland chromatin 8, normal nucleoli 10, mitoses 1 was classified as malignant?
```

- Credit:
```
Hello, my credit request information is as following: a, 22.67, 0.750, u, g, c, v, 2.00, f, t, 2, t, g, 200.0, 394. Why my loan was rejected?
```

# Citing the paper

If MAINLE was useful for you, please consider citing the paper.

## BibTex

```
@inproceedings{serafim2025mainle,
  author = {Serafim, Paulo Bruno and
    F\'{e}rrer Filho, R\^{o}mulo and
    Freitas, Stenio and
    Gezici, Gizem and
    Giannotti, Fosca and
    Raimondi, Franco and
    Santos, Alexandre},
  title = {{MAINLE}: A Multi-Agent, Interactive, Natural Language Local Explainer of Classification Tasks},
  booktitle = {Machine Learning and Knowledge Discovery in Databases. Research Track: European Conference, ECML PKDD 2025, Porto, Portugal, September 15--19, 2025, Proceedings, Part IV},
  publisher = {Springer-Verlag},
  address = {Berlin, Heidelberg},
  pages = {149--165},
  numpages = {17},
  year = {2025},
  isbn = {978-3-032-06077-8},
  doi = {10.1007/978-3-032-06078-5_9}
}
```
