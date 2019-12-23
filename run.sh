#!/bin/bash


usage() {
    echo "Usage: $0 [-s] [-e]"
    echo "       -s  : Runs Swap Random Forest (Only swap mechanism)"
    echo "       -e  : Runs Elastic Adaptive Random Forest (Only elastic mechanism)"
    echo ""
    echo " If no option specified: Runs Elastic Swap Random Forest (both elastic and swap mechanism)"
    echo ""
    1>&2; exit 1;
}


CLASSPATH=target/Crunchify/Crunchify.jar
DS=datasets/elecNormNew.arff
SAMPLING_FREQUENCY=100000
PERF_EVAL=BasicClassificationPerformanceEvaluator



run_moa(){
java -cp $CLASSPATH  moa.DoTask $1 
}

evaluate_preqcv(){
    run_moa "EvaluatePrequentialCV -l ($1)  -s (ArffFileStream -f $DS) -e $PERF_EVAL -f $SAMPLING_FREQUENCY" 
}



run_swap_only(){
    echo ""
    echo "·===[ EVALUATING SWAP RANDOM FOREST ]====================================================·"
    evaluate_preqcv "meta.SwapRandomForest -f 10 -c 10 -s 100"
}


run_elastic_only(){
    echo ""
    echo "·===[ EVALUATING ELASTIC ADAPTIVE RANDOM FOREST ]====================================================·"
    evaluate_preqcv "meta.ElasticARF -f 10 -c 10 -s 100"
}

run_elastic_swap(){
    echo ""
    echo "·===[ EVALUATING ELASTIC SWAP RANDOM FOREST ]====================================================·"
    evaluate_preqcv "meta.ElasticRandomForest -f 10 -c 10 -s 100"
}





SWAP_ONLY=0
ELASTIC_ONLY=1
ELASTIC_SWAP=2



RUN_OPT=$ELASTIC_SWAP


while getopts "d:es" o; do
    case "${o}" in
	d)
	    DS=$OPTARG
	    ;;

        e)
	    RUN_OPT=$ELASTIC_ONLY
	    
	    #SWAP_ONLY=0
	    #ELASTIC_ONLY=1

	    #run_elastic_only
	    #exit 0
            ;;
        s)
	    RUN_OPT=$SWAP_ONLY

	    #SWAP_ONLY=1
	    #ELASTIC_ONLY=0
	    
            #run_swap_only
	    #exit 0
            ;;

        *)
            usage
            ;;
    esac
done
shift $((OPTIND-1))



case "${RUN_OPT}" in
    $SWAP_ONLY)
	run_swap_only
        ;;
    $ELASTIC_ONLY)
	run_elastic_only
        ;;
    $ELASTIC_SWAP)
	run_elastic_swap
        ;;

esac




