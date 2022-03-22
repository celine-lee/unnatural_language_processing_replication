# Midterm Replication Project: Unnatural Language Processing: Bridging the Gap Between Synthetic and Natural Language Data

## Introduction

## Method

## Results



## How to replicate


### Setup

```
conda create --name synthetic python=3.8
conda activate synthetic

conda install pytorch
pip install -U sentence-transformers
```

### Run
```
python seq2seq.py --optimizer adam --epochs 20 --batch_size 8 --save_model --train_model --model_filename "seq2seq_20eps"
python baseline.py --batch_size 8 --model_filename seq2seq_20eps --output_filename 'output/seq2seq_20eps_projection.txt'
```




## nevermind

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