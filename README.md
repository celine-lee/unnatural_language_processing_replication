# Midterm Replication Project: Unnatural Language Processing: Bridging the Gap Between Synthetic and Natural Language Data

['Unnatural Language Processing: Bridging the Gap Between Synthetic and Natural Language Data' by Alana Marzoev, Samuel Madden, M. Frans Kaashoek, Michael Cafarella, Jacob Andreas](https://arxiv.org/pdf/2004.13645.pdf)


## Introduction
The objective of the paper is to generate and train a model on synthetic text data to perform some natural language processing task. This paper builds off of previous work (['Building a Semantic Parser Overnight'](https://aclanthology.org/P15-1129.pdf)) that uses a program and synthetic utterance grammar to generate corresponding canonical synthetic utterances for each program. The contribution of this paper is the development of a *projection model* to map natural language utterances to sentences in the synthetic text domain. The synthetic sentence can then be used as input to the trained semantic parser.

The combination of the (1) synthetic data generation, (2) trained synthetic data task model, and (3) projection model compose a framework for developing NLP systems without large annotated datasets.

At a high level, the pipeline looks as follows:
![Pipeline](images/system_pipeline.png?raw=true "system pipeline diagram")

In this replication, we focus on this synthetic model and projection framework in the context of Semantic Parsing with the Overnight dataset.

## Method
To replicate this part of this paper, I broke the process down into three primary components: (1) collect synthetic data, (2) train semantic parser on synthetic utterances and programs, (3) write projection model from NL utterances to synthetic utterances in the grammar. 

### (1) Collect synthetic data
Data is collected from the Calendar domain of the Overnight dataset. 

Download data files from the SEMPRE CodaLab: [worksheet](https://worksheets.codalab.org/worksheets/0x269ef752f8c344a28383240f7bb2be9c)
Files are collected in the `overnight_data` folder.

(The SEMPRE full semantic parsing pipeline, which we do not use but is the resulting model from the original Overnight dataset paper, can also be run in an interactive shell by following the [instructions](#(unused,-but-interesting)-run-stanford-semantic-parsing-model-(trained-on-overnight-dataset))

Once the data files for grammar and examples were downloaded, I wrote a dataloader for the original synthetic canonical utterance, natural language paraphrase, and Calendar domain program. Based on whether I want to train on only synthetic->program, paraphrase->program, synthetic/paraphrase->program, I load the corresponding input utterance and output programs accordingly. 

### (2) Train semantic parser
With the dataloader written, I follow suit of the paper and train a LSTM encoder-decoder model with an embedding size of 256 and a hidden size of 1024. This model is the seq2seq semantic parser from utterance to program in the Calendar domain. 

Graphs from the training procedure on training the models on synthetic data, NL paraphrases, and both are shown below:
![Training Performance](images/train_performance.png?raw=true "seq2seq training performance")
![Test Performance](images/test_performance.png?raw=true "seq2seq test performance")

### (3) Write projection model
In the paper, for all datasets in the semantic parsing Overnight dataset, the authors use the "flat" projection model <img src="https://render.githubusercontent.com/render/math?math=\pi">, which is to select the "closest" synthetic utterance to an input natural language utterance by using sentence embeddings (embed) and searching over all synthetic utterances <img src="https://render.githubusercontent.com/render/math?math=\tilde{x}"> in a set of utterances <img src="https://render.githubusercontent.com/render/math?math=\widetilde\mathcal{X}"> to find the one with the closest cosine similarity <img src="https://render.githubusercontent.com/render/math?math=\delta">:
![Projection](images/projection_eqn.png?raw=true "flat projection formula")

Therefore, we do the same, using SentenceTransformers's 'all-MiniLM-L12-v2' model and sklearn's cosine_similarity metric. The paper is unclear on how they obtain the set of synthetic utterances <img src="https://render.githubusercontent.com/render/math?math=\widetilde\mathcal{X}">, so I tried a couple techniques, all documented below:
- save all synthetic utterances from training into <img src="https://render.githubusercontent.com/render/math?math=\widetilde\mathcal{X}">
- save all synthetic utterances from training and test into <img src="https://render.githubusercontent.com/render/math?math=\widetilde\mathcal{X}">
- save all synthetic utterances from training and programmatically augment each example with argument flips of matching type, according to the grammar files

## Results
| Model         | Train data | Program Accuracy | (w test) | (w augment) | (w test & augment) | Synth Utterance Accuracy | (w test)  | (w augment) | (w test & augment)| Paper Accuracy|
| ------------- |:----------:|:----------------:|:--------:|:-----------:|:------------------:|:------------------------:|:---------:|:-----------:|:-----------------:|:-------------:|
| adagrad-8eps  | synthetic  | 0.0              | 0.0      | 0.0         | 0.0                | 0.4166                   |  0.4226   | 0.3988      | 0.3988            |   0.32        |
| adam-20eps    | synthetic  | 0.4167           | 0.4226   | --          | --                 | 0.4167                   |  0.4226   |  --        | --                 | 0.32          |
| adam-8eps     | synthetic  | 0.4167           | 0.4226   | 0.3988      | 0.3988             | 0.4167                   |  0.4226   | 0.3988     | 0.3988             | 0.32          |
| adam-15-real  | real       | 0.0119           | 0.0119   | 0.0119      | 0.0119             | 0.4167                   |  0.4226   | 0.3988     | 0.3988             | --          |
| adam-15-both  | both       | 0.0              | 0.0      | 0.0         | 0.0                |  0.4167                  |  0.4226   | 0.3988     | 0.3988             | --          |
| adam-15-real  | real       | (no projection) 0.5298 | -- | --          | --                 | --                       |  --       | --         | --                 | 0.27          |
| adam-15-both  | both       | (no projection) 0.0179 | -- | --          | --                 | --                       |  --       | --         | --                 | 0.13          |

My results follow the same general model performance trend: high performance with the synthetic data and projection model ("synthetic + projection") and the real data with no projection model ("real"), and low performance with both real and synthetic data with no projection model ("both"). However, the results did vary somewhat significantly. I discuss some theories and experiments around the variance below:

- Performance of my "synthetic + projection" model outperformed that of the paper. I believe this may have to do with the smaller set of synthetic utterances I was projecting onto, as my seq2seq model was fit to the set of synthetic utterances I mapped onto, and my program accuracy matched my synthetic utterance match accuracy. To test this, I artificially "augmented" the set of synthetic utterances: for each synthetic utterance, create additional synthetic utterances by swapping out arguments for other arguments of the same semantic type from the grammar. I noticed that performance dropped with augmentation, which suggest support for my theory about performance increase due to overfitting and bias with a smaller set of synthetic utterances.
- Performance of my "real" model outperformed their "real" model by almost double the accuracy. I assume this is either a bug in my code or there was some inconsistency in model, number of training epochs, data set, or other. 
- Performance of my "both" model underperformed their "both" model by almost 10x less. I assume this is either a bug in my code or there was some inconsistency in model, number of training epochs, data set, or other. 


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
        --output_file 'output/seq2seq_adam_8_projection.txt'
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