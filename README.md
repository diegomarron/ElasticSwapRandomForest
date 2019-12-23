# ElasticSwapRandomForest
The Resource-aware Elastic Swap Random Forest for Evolving Data Streams (ESRF) algorithm reduces the number of trees in the Random Forest up to one third on average while providing the same accruacy.

This repository contains two orthogonal components:
+ **Swap component**: splits learners into two sets based on their accuracy
+ **Elastic component**: dynamically increases/decreases the number of learners in the ensemble


Files:
+ *src/main/java/moa/classifiers/meta/SwapRandomForest.java*: Only the Swap component implemented on top of the Adaptive Random Forest
+ *src/main/java/moa/classifiers/meta/ElasticRandomForest.java*: Both Swap and Elastic components implemented on top of the Adaptive Random Forest
+ *src/main/java/moa/classifiers/meta/ElasticARF.java*: Only the elastic component only  implemented on top of the Adaptive Random Forest



## Citing Elastic Swap Random Forest

For more details, please refer to the following publication:
> [PDF on Arxiv](https://arxiv.org/pdf/1905.05881.pdf)

To cite this project in a publication, please cite the following paper:
> D Marrón, E Ayguadé, JR Herrero, A Bifet
> Resource-aware Elastic Swap Random Forest for Evolving Data Streams

Or directly copy the bibtext:
```
@misc{marrn2019resourceaware,
    title={Resource-aware Elastic Swap Random Forest for Evolving Data Streams},
    author={Diego Marrón and Eduard Ayguadé and José Ramon Herrero and Albert Bifet},
    year={2019},
    eprint={1905.05881},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```


## Clone repository and build


**This repo REQUIRES MAVEN installed**
[How to install Maven](https://maven.apache.org/install.html)


This repository only contains the contributions explained in the above paper. We make use of maven to download moa artifact and create a jar file containing everithing.

Complete jar relative path: target/Crunchify/Crunchify.jar



```
git clone https://github.com/diegomarron/ElasticSwapRandomForest
cd ElasticSwapRandomForest

mvn package

```

## How to execute it

### Using run.sh script

The easiest way to run a quick experiment is by using the run.sh scrip provided using the elecNormnew dataset (included in dataset/elecNormnew.arff).


To run the Swap only mechanism (SwapRandomforest)
```
bash run -s

```

If no option is given, the scrip run the Swap and Elastic mechanism (ElasticSwapRandomforest)
```
bash run

```

To run the Elastic only mechanism (Elastic AdaptiveRandomforest)
```
bash run -e

```


To test using a different dataset, use the -d /PATH/TO/DATASET. For example, using the elecnormnew dataset included in this repo:
```
bash run -e -d datasets/elecNormNew.arff

```



### Run Manually from command line



#### Swap Random Forest

```
java -cp target/Crunchify/Crunchify.jar moa.DoTask 'EvaluatePrequentialCV -l (meta.SwapRandomForest -f 10 -c 10 -s 100) -s (ArffFileStream -f datasets/elecNormNew.arff)) -e BasicClassificationPerformanceEvaluator -f 100000'

```


#### Elastic Swap Random Forest

```
java -cp target/Crunchify/Crunchify.jar moa.DoTask 'EvaluatePrequentialCV -l (meta.ElasticRandomForest -f 10 -c 10 -s 100) -s (ArffFileStream -f datasets/elecNormNew.arff)) -e BasicClassificationPerformanceEvaluator -f 100000'

```

#### Elastic Adaptive Random Forest

```
java -cp target/Crunchify/Crunchify.jar moa.DoTask 'EvaluatePrequentialCV -l (meta.ElasticARF -f 10 -c 10 -s 100) -s (ArffFileStream -f datasets/elecNormNew.arff)) -e BasicClassificationPerformanceEvaluator -f 100000'

```







