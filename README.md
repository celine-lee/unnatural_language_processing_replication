# Midterm Replication Project: Unnatural Language Processing: Bridging the Gap Between Synthetic and Natural Language Data

['Unnatural Language Processing: Bridging the Gap Between Synthetic and Natural Language Data' by Alana Marzoev, Samuel Madden, M. Frans Kaashoek, Michael Cafarella, Jacob Andreas](https://arxiv.org/pdf/2004.13645.pdf)


## Introduction
The task of the paper is to generate and train a model on synthetic text data to perform some natural language processing task. This paper builds off of the use of a program and synthetic utterance grammar to generate corresponding canonical synthetic utterances for each program, a technique presented in previous works, one prominent one of which is ['Building a Semantic Parser Overnight'](https://aclanthology.org/P15-1129.pdf). The contribution of this paper is the development of a projection model to map natural language utterances to sentences in the synthetic text domain. The synthetic sentence can then be used as input to the trained semantic parser.

The combination of the (1) synthetic data generation, (2) trained synthetic data task model, and (3) projection model compose a framework for developing NLP systems without large annotated datasets.

At a high level, the pipeline looks as follows:
![Pipeline](images/system_pipeline.png?raw=true "system pipeline diagram")

In this replication, we focus on their framework in the context of Semantic Parsing with the Overnight dataset.

## Method
To replicate this part of this paper, we break the process down into three primary components: (1) collect synthetic data, (2) train semantic parser on synthetic utterances and programs, (3) write projection model from NL utterances to synthetic utterances in the grammar. 

### (1) Collect synthetic data
tODO

### (2) Train semantic parser
TODO
Graphs from the training procedure are shown below:
![Training Accuracy](images/accuracy_plot.png?raw=true "seq2seq training accuracy")
![Training Loss](images/loss_plot.png?raw=true "seq2seq training loss")
![Test Accuracy](images/test_acc_plot.png?raw=true "seq2seq test accuracy")
![Test Loss](images/test_loss_plot.png?raw=true "seq2seq test loss")

### (3) Write projection model
TODO

## Results
| Model         | Projector | Program Accuracy | Program Accuracy (w test)| Synth Utterance Accuracy | Synth Utterance Accuracy (w test) | Paper Accuracy|
| ------------- |:---------:|:----------------:|:-----------------------:|:------------------------:|:---------------------------------:|:-------------:|
| adagrad-8eps  | Flat      | 0.0              | 0.0                     | 0.4166                   |  0.4226                           |   0.32        |
| adam-20eps    | Flat      | 0.4167           | 0.4226                  | 0.4167                   |  0.4226                           | 0.32          |
| adam-8eps     | Flat      | 0.4167           | 0.4226                  | 0.4167                   |  0.4226                           | 0.32          |

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
        --epochs 8 \
        --batch_size 8 \
        --save_model --train_model \
        --model_filename 'seq2seq_adam_8'
```

Once the (synthetic) semantic parser has been trained, we pre-pend a projection model. The projection model projects natural language inputs to utterances in the synthetic grammar, which can be fed as input to the trained semantic parser. The following command loads the trained semantic parsing model, runs a projection model with the semantic parsing model on all natural language paraphrases from the test dataset, and calculates the exact-match accuracy of the results.
```
python baseline.py \
        --batch_size 8 \
        --model_filename 'seq2seq_adam_8' \
        --output_filename 'output/seq2seq_adam_8_projection.txt'
```

An interactive version of the interface can be run with the following command:
```
python baseline_demo.py \
        --model_filename 'seq2seq_adam_8'
        --output_file 'output/interactive_projection.txt'
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