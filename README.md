# How to install the library #

### 1. Download all the requirements before running: ###

   ```bash
    pip install -r requirements.txt
   ```
Note: pygraphviz need extra step to download, please see the suggested guidance as following.

 ### 2. Downaload Graphviz and pygraphviz ###

In edge attribution patching part, it need to use pygraphviz

#### Ubuntu/Debian ####

```bash
sudo apt-get update
sudo apt-get install -y graphviz libgraphviz-dev
pip install pygraphviz
```

#### macOS ####

```bash
brew install graphviz
export CFLAGS="-I$(brew --prefix graphviz)/include"
export LDFLAGS="-L$(brew --prefix graphviz)/lib"
pip install pygraphviz
```
#### Windows ####

1. download Graphviz from https://graphviz.org/download/
2. Add the bin directory of Graphviz into system PATH.
3. run:
   ```bash
   pip install pygraphviz
   ```


**Dataset**

This project loads "yelp_polarity" and "stanfordnlp/imdb" datasets from huggingface automatically.
For quick trial runs and demonstrations, this  'data' directory contains sample data from the original dataset and some corrupted data samples for yelp dataset.
