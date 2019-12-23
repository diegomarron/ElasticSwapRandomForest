# ElasticSwapRandomForest
The Resource-aware Elastic Swap Random Forest for Evolving Data Streams (ESRF) algorithm reduces the number of trees in the Random Forest up to one third on average while providing the same accruacy.


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

** REQUIRES MAVEN **

```
git clone https://github.com/diegomarron/ElasticSwapRandomForest
cd ElasticSwapRandomForest

mvn package

```

## How to execute it


### Swap Random Forest

```
java -cp target/Crunchify/Crunchify.jar moa.DoTask 'EvaluatePrequentialCV -l (meta.SwapRandomForest -f 10 -c 10 -s 100) -s (ArffFileStream -f /PATH/TO/ARFF_FILE) -e BasicClassificationPerformanceEvaluator -f 100000'

```


### Elastic Swap Random Forest

```
java -cp target/Crunchify/Crunchify.jar moa.DoTask 'EvaluatePrequentialCV -l (meta.ElasticRandomForest -f 10 -c 10 -s 100) -s (ArffFileStream -f /PATH/TO/ARFF_FILE) -e BasicClassificationPerformanceEvaluator -f 100000'

```

### Elastic Adaptive Random Forest

```
java -cp target/Crunchify/Crunchify.jar moa.DoTask 'EvaluatePrequentialCV -l (meta.ElasticARF -f 10 -c 10 -s 100) -s (ArffFileStream -f /PATH/TO/ARFF_FILE) -e BasicClassificationPerformanceEvaluator -f 100000'

```







