# CCMAS16
MFMC is an acronym that stands for ‘MIDI From Multiple Contents’ and represents a group project created by four students at the University of Helsinki. This project is written entirely using Python 3.5, utilizes the Creamas multi-agent library atop of aiomas, the NLTK language processing toolkit, and creates its MIDI files using the Pyknon library. It supports creating a user-defined number of agents which are assigned reading material from the selection of TXT files placed in the InspiringSet subfolder. After agents have sanitized the material, they will create Markov Chains based on word probability of a user-defined order.

The agents then use these Markov Chain probabilities to construct new sentences, which they then treat as lyrics to a song. Each agent creates their own set of ‘tools’ which combined are called an ‘invention method’. They then create music according to the instructions derived from their choice of invention method, other agents rate their results, and they apply this feedback to the tools used so they can be ranked by effectiveness. Agents will continue to create new music every round, continually evaluating their individually unique methods and revising or discarding them based on the feedback received from their peers.

## Installation and Usage
MFMC can be executed by installing Python 3.5, using pip to install the following libraries:

* [NLTK](http://www.nltk.org/) `pip install nltk`
* [Pyknon](https://github.com/kroger/pyknon) `pip install pyknon`
* [Creamas](https://pypi.python.org/pypi/creamas/0.1.0) `pip install creamas`
* [NumPy](http://www.numpy.org/) `pip install numpy`

Cloning the project to a local folder:

`git clone git@github.com:cabrill/CCMAS16.git`

And then executing the mfmc.py file from the folder you have cloned MFMC to from Github:

`python mfmc.py`

All command line arguments are optional:

```
usage: mfmc.py [-h] [-a AGENTS] [-p PATH] [-o ORDER] [-r ROUNDS] [-m MEMORY]

optional arguments:

  -a AGENTS, --agents AGENTS
                        The number of concurrent agents to simulate.
  -p PATH, --path PATH  The location of the inspiring set of TXT files.
  -o ORDER, --order ORDER
                        The order of Markov Chain to use in lyric generation.
  -r ROUNDS, --rounds ROUNDS
                        The number of voting rounds to simulate.
  -m MEMORY, --memory MEMORY
                        The number of previous artifacts an agent can
                        remember.
```