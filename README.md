# Midterm Replication Project: Unnatural Language Processing: Bridging the Gap Between Synthetic and Natural Language Data

## Introduction

## Method


![Training Accuracy](images/accuracy_plot.png?raw=true "seq2seq training accuracy")
![Training Loss](images/loss_plot.png?raw=true "seq2seq training loss")

## Results
| Model         | Projector | Accuracy (EM)| Paper Accuracy|
| ------------- |:---------:|:------------:|:-------------:|
| adagrad-8eps  | Flat      | 0.0          | 0.32          |
| adam-20eps    | Flat      | 0.25         | 0.32          |

## How to replicate


### Setup

```
conda create --name synthetic python=3.8
conda activate synthetic

conda install pytorch
conda install -c conda-forge wandb 
pip install -U sentence-transformers
```

### Run
First, we need to train a semantic parser from the synthetic utterances to programs in the Overnight grammar. Customize the optimizer, number of epochs, batch size, and model filename as desired:
```
python seq2seq.py \
        --optimizer adam \
        --epochs 20 \
        --batch_size 8 \
        --save_model --train_model \
        --model_filename "seq2seq_20eps"
```

Once the (synthetic) semantic parser has been trained, we pre-pend a projection model. The projection model projects natural language inputs to utterances in the synthetic grammar, which can be fed as input to the trained semantic parser. The following command loads the trained semantic parsing model, runs a projection model with the semantic parsing model on all natural language paraphrases from the test dataset, and calculates the exact-match accuracy of the results.
```
python baseline.py \
        --batch_size 8 \
        --model_filename seq2seq_20eps \
        --output_filename 'output/seq2seq_20eps_projection.txt'
```

An interactive version of the interface can be run with the following command:
```
TODO
```


### (Unused, but interesting) Run Stanford Semantic Parsing Model (Trained on Overnight dataset)

Download [Apache Ant](https://ant.apache.org/manual/install.html)
`ANT_HOME=/Users/celinelee/Downloads/apache-ant-1.9.16` or wherever you uncompressed the download to.
`PATH=${ANT_HOME}/bin:$PATH`

Install [Java JDK](https://www.oracle.com/java/technologies/downloads/#jdk17-mac).

```
git clone https://github.com/percyliang/sempre
cd sempre
ruby ./pull-dependencies core
ant core
ruby ./run @mode=simple
```

Useful: https://github.com/percyliang/sempre/blob/master/TUTORIAL.md