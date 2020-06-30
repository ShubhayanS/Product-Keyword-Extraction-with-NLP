[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)
[![License](http://img.shields.io/:license-gnu-blue.svg)](http://doge.gnu-license.org)
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/ansicolortags.svg)](https://pypi.python.org/pypi/ansicolortags/)
![Build Status](http://img.shields.io/travis/badges/badgerbadgerbadger.svg?style=flat-square)
[![Open Source Love png1](https://badges.frapsoft.com/os/v1/open-source.png?v=103)](https://github.com/ellerbrock/open-source-badges/)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](http://makeapullrequest.com)
[![GitHub stars](https://img.shields.io/github/stars/ShubhayanS/Product-Keyword-Extraction-with-NLP?style=social&label=Star&maxAge=2592000)](https://GitHub.com/ShubhayanS/Product-Keyword-Extraction-with-NLP/stargazers/)
[![GitHub forks](https://img.shields.io/github/forks/ShubhayanS/Product-Keyword-Extraction-with-NLP?style=social&label=Fork&maxAge=2592000)](https://GitHub.com/ShubhayanS/Product-Keyword-Extraction-with-NLP/network/)
# Product-Keyword-Extraction-with-NLP

In this repo, NER is used to extract keywords from a large text. Named-entity recognition (NER) is the process of automatically identifying the entities discussed in a text and classifying them into pre-defined categories such as ‘person’, ‘organization’, ‘location’ and so on. The spaCy library allows you to train NER models by both updating an existing spacy model to suit the specific context of your text documents and also to train a fresh NER model from scratch.

Why not regex or any other NLP technique other than using Spacy?
This is because using regex its generally slower and instead of extracting particular keywords lot of junk irrelevant keywords are also identified.Moreover with superfast and efficiency of 90% Spacy is most widely used in production level. Read about Spacy at https://spacy.io/

### Installation and Setup

- Requirements for running this NLP model is Python along with Spacy. We first load the small pretrained spacy model but larger and more detailed model can be loaded by changing to "lg" instead of "sm" and then we train our custom NER (Named Entity Recognisation) for fine tuning and detection of E-Commerce products like "Alto" or "Dell Laptop". To understand each code modules we have model.py where model is getting saved and test.py for inputing custom text and testing.

> Install pip for Python3

```shell
$ sudo apt-get upgrade
$ sudo apt install python3-pip
```

> now install spacy and small model packages

```shell
$ pip install spacy
$ python -m spacy download en_core_web_sm
```


## Contributing

> To get started...

### Step 1

- **Option 1**
    - 🍴 Fork this repo!

- **Option 2**
    - 👯 Clone this repo to your local machine using `https://github.com/ShubhayanS/Product-Keyword-Extraction-with-NLP.git`

### Step 2

- **HACK AWAY!** 🔨🔨🔨

### Step 3

- 🔃 Create a new pull request using <a href="https://github.com/ShubhayanS/Product-Keyword-Extraction-with-NLP/compare/" target="_blank">`https://github.com/ShubhayanS/Product-Keyword-Extraction-with-NLP/compare/`</a>.
