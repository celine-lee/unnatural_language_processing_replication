## Setup

```
conda create --name synthetic python=3.8
conda activate synthetic

conda install pytorch
```


## Run
```
python seq2seq.py --train_model
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