/*
 *    ElasticSwapRandomForest.java
 * 
 *    @author Diego Marrón Vida 
 *
 *    Licensed under the Apache License, Version 2.0 (the "License");
 *    you may not use this file except in compliance with the License.
 *    You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *    Unless required by applicable law or agreed to in writing, software
 *    distributed under the License is distributed on an "AS IS" BASIS,
 *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *    See the License for the specific language governing permissions and
 *    limitations under the License.
 *
 */

package moa.classifiers.meta;

import com.yahoo.labs.samoa.instances.Instance;

import moa.classifiers.AbstractClassifier;
import moa.classifiers.MultiClassClassifier;
import moa.core.DoubleVector;
import moa.core.InstanceExample;
import moa.core.Measurement;
import moa.core.MiscUtils;
import moa.options.ClassOption;
import moa.core.Example;
import moa.core.Utils;


import com.github.javacliparser.FloatOption;
import com.github.javacliparser.FlagOption;
import com.github.javacliparser.IntOption;
import com.github.javacliparser.MultiChoiceOption;
import java.util.ArrayList;
import java.util.Collection;
import java.util.concurrent.Callable;
import java.util.Random;
import java.util.Arrays;

import moa.classifiers.trees.ARFHoeffdingTree;
import moa.evaluation.BasicClassificationPerformanceEvaluator;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import moa.classifiers.core.driftdetection.ChangeDetector;
import java.text.DecimalFormat;

import java.io.Serializable;


public class SwapRandomForest extends AbstractClassifier implements MultiClassClassifier {


    /////////////////////////////////////////////////////////////////////////////////////
    //
    /// ELASTIC BASIC ACCURACY KEEPER
    //
    /////////////////////////////////////////////////////////////////////////////////////

    protected class BasicAccuracy{

	//private static final long serialVersionUID = 1L;

	protected long[] m_classified = {0,0};
	protected static final int s_OK=0;
	protected static final int s_WRONG=1;

	public void reset() {
	    this.m_classified[0]=0;
	    this.m_classified[1]=0;
	}

	public void set(BasicAccuracy ba) {
	    this.m_classified[0]=ba.m_classified[0];
	    this.m_classified[1]=ba.m_classified[1];
	}

	
	public void addResult(int trueClass, double[] votes){
	    
            int predictedClass = Utils.maxIndex(votes);

	    int idx=(trueClass == predictedClass) ? s_OK : s_WRONG;
	    this.m_classified[idx]++;
	}


	public void addResult(int trueClass, int predictedClass){
	    
            //int predictedClass = Utils.maxIndex(votes);

	    int idx=(trueClass == predictedClass) ? s_OK : s_WRONG;
	    this.m_classified[idx]++;
	}

	
	
	/*
	public void addResult(Instance instance, double[] votes){

	    if (instance.classIsMissing() == true)
		return;

	    double weight = instance.weight();
	    if (weight == 0.0)
		return;

	    int trueClass = (int) instance.classValue();
            int predictedClass = Utils.maxIndex(votes);

	    int idx=(trueClass == predictedClass) ? s_OK : s_WRONG;
	    this.m_classified[idx]++;
	}
	*/
	
	public double get(){
	    double T=this.m_classified[s_OK] + this.m_classified[s_WRONG];
	    return (this.m_classified[s_OK] / T);
	}
    }


    /////////////////////////////////////////////////////////////////////////////////////
    //
    //// 2x2 CONTINGENCY TABLE
    //
    /////////////////////////////////////////////////////////////////////////////////////

    protected class ContingencyTable{

	//private static final long serialVersionUID = 1L;

	protected long[][] m_table;
	protected long m_total;
	//protected double m_accum;
	
	public ContingencyTable(){
	    this.m_table = new long[2][2];
	    reset();
	}

	public void reset(){
	    this.m_table[0][0]=0;
	    this.m_table[0][1]=0;
	    this.m_table[1][0]=0;
	    this.m_table[1][1]=0;
	    this.m_total=0;
	    //this.m_accum=0;
	}
	
	
	
	public void addResult(int targetLabel,
			      long casePrediction,
			      long controlPrediction){

	    int isCasePositive = (casePrediction==targetLabel)?  0 : 1 ;	    
	    int isControlPositive = (controlPrediction==targetLabel)?  0 : 1;	    
	    /*
	    System.out.println("[ADDRESULT] label> "
			       + " target:"+targetLabel
			       + " case:"+casePrediction
			       + " controlPrediction:"+controlPrediction
			       + " | IDX:[" + isControlPositive
			       + "][" + isCasePositive + "]"
			       );
	    */
	    
	    //m_table[isCasePositive][isControlPositive]++;
	    m_table[isControlPositive][isCasePositive]++;
	    this.m_total++;
	}

	
	//public long[2][2] getTable(){ return m_table}

	public double accuracyCase(){
	    
	    double n = this.m_table[0][0]+this.m_table[0][1];
	    return n/this.m_total;
	}

	public double accuracyControl(){
	    
	    double n = this.m_table[0][0]+this.m_table[1][0];
	    return n/this.m_total;
	}

	
	
	public double mcnemar(){
	    //https://www.graphpad.com/quickcalcs/McNemar1.cfm
	    /*
	    System.out.println("[CONTINGENCY] \n"
			       + this.m_table[0][0] + ", "  + this.m_table[0][1] +"\n"
			       + this.m_table[1][0] + ", "  + this.m_table[1][1] );
	    */
	    
	    double b = this.m_table[0][1];
	    double c = this.m_table[1][0];

	    double den = b + c;
	    double num=(b-c);


	    //e mid P-value

	    /*
	    if(den<6){
		// use mid P-value
		num=Math.abs(num)-1;
		num*=num;
		return (num/den);
		
	    }
	    */

	    if (den==0)
		return 0;

	    
	    num=Math.abs(num)-0.5;
	    num*=num;
	    return (num/den);

	    
	    /*
	    if (den<25){

	    // Yates's continuity correction
	    //num=Math.abs(num)-0.5;

	    // Edward's mcnemar continuity correction
	    // to aprox binomial exact P-value
		num=Math.abs(num)-1;
		num*=num;
		return (num/den);
	    }else{
		num=num*num;
		num*=num;
		return (num/den);
	    }
	    */
		
	    /*
	    if (den<25){


		// Edward's mcnemar continuity correction
		// to aprox binomial exact P-value
		num=Math.abs(num)-1;
		num*=num;
		double edward=(num/den);


		double bin=
		return (num/den);

	    }else{
		// mcnemar continuity correction
		//num=Math.abs(num)-1;
		num=num*num;
		num*=num;
		return (num/den);

		//double num=(b-c);
		//num*=num;
		//num*=num;
		//return (num/den);
	    }
	    */
	}
	
    }
    /*    
    class LearnerStatistics{
	public int indexOriginal;
        public long createdOn;
        public long lastDriftOn;
        public long lastWarningOn;
	public long instancesSeen;

	//ContingencyTable 
	
	public LearnerStatistics(){
	    this.reset();
	}

	public void reset(){
	    this.indexOriginal=0;
	    this.createdOn=0;
	    this.lastDriftOn=0;
	    this.lastWarningOn=0;
	    this.instancesSeen=0;

	}
	
    }
    */
    /*

    class ElasticLearner
    {
    }


    
    class LearnerContainer<T extends C>
    {
	
    }
    */
    /*
    class learnerFactory{
	
    }
    

    class learnerAllocator<T>{
	protected ClassOption m_treeLearnerOption;

	protected T m_learnerZygote;
	
	public learnerAllocator(ClassOption opt){
	    this.m_treeLearnerOption = opt;
	    this.m_learnerZygote=(T)getPreparedClassOption(this.m_learnerClassOption);
	}

	public T alloc(){
	    return (T) this.m_learnerZygote.copy();
	}
    }

    
    class ElasticLearner{

	public ARFHoeffdingTree learner;
	
	public ElasticBaseLearner(int indexOriginal,
				  long instancesSeen){

	    this.learner=learnerAllocator
	    
	}
				  

    }

    */

    /////////////////////////////////////////////////////////////////////////////////////
    //
    //// ELASTIC RANDOM FOREST CONFIRURATION
    //
    /////////////////////////////////////////////////////////////////////////////////////

    protected class ElasticConfig{

	//private static final long serialVersionUID = 1L;

	protected double m_lambda;
	protected int   m_candidatesSize;
	protected int   m_frontSize;
	protected int   m_maxSize;
	
	ElasticConfig(FloatOption lambdaOption,
		      IntOption candidateSizeOption,
		      IntOption frontSizeOption,
		      IntOption maxSizeOption){
	    this.m_lambda = lambdaOption.getValue();
	    this.m_candidatesSize= candidateSizeOption.getValue();
	    this.m_frontSize=frontSizeOption.getValue();
	    this.m_maxSize=maxSizeOption.getValue();
	}

	double getLambda() { return this.m_lambda; }
	int getCandidatesSize() { return this.m_candidatesSize; }
	int getFrontSize() { return this.m_frontSize; }
	int getLearnersMaxSize() { return this.m_maxSize; }
    }
    
    /////////////////////////////////////////////////////////////////////////////////////
    //
    //// ELASTIC LEARNER STATICTICS
    //
    /////////////////////////////////////////////////////////////////////////////////////

    class LearnerStatistics{

	//private static final long serialVersionUID = 1L;

	public int indexOriginal;
	public long createdOn;
	public long lastDriftOn;
	public long lastWarningOn;
	public long instancesSeen;

	//ContingencyTable 
	
	public LearnerStatistics(){
	    this.reset();
	}

	public void reset(){
	    this.indexOriginal=0;
	    this.createdOn=0;
	    this.lastDriftOn=0;
	    this.lastWarningOn=0;
	    this.instancesSeen=0;

	}
	
    }

    
    /////////////////////////////////////////////////////////////////////////////////////
    //
    //// Elastic Learner Wrapper
    //
    /////////////////////////////////////////////////////////////////////////////////////
    
    class ElasticBaseLearner{ // extends ARFBaseLearner{

	//private static final long serialVersionUID = 1L;
	
	public int indexOriginal;
	public long createdOn;
	public long lastDriftOn;
	public long lastWarningOn;
	public long instancesSeen;

	public ARFHoeffdingTree classifier;
        //public BasicClassificationPerformanceEvaluator evaluator;
	public BasicAccuracy accuracy;

	
        protected ClassOption driftOption;
	protected ChangeDetector driftDetectionMethod;

	
	public ElasticBaseLearner(int indexOriginal,
				  ARFHoeffdingTree instantiatedClassifier,
				  BasicClassificationPerformanceEvaluator evaluatorInstantiated, 
				  long instancesSeen,
				  boolean useBkgLearner,
				  boolean useDriftDetector,
				  ClassOption driftOption,
				  //ClassOption warningOption,
				  boolean isBackgroundLearner) {

	    
            this.indexOriginal = indexOriginal;
            this.createdOn = instancesSeen;
            this.lastDriftOn = 0;
            this.lastWarningOn = 0;

	    this.classifier = instantiatedClassifier;
	    //this.evaluator = evaluatorInstantiated;

	    this.accuracy = new BasicAccuracy();
	    
	    this.driftOption = driftOption;
	    this.driftDetectionMethod=((ChangeDetector)getPreparedClassOption(this.driftOption)).copy();

	    this.instancesSeen=instancesSeen;
	}

	

	public void reset(){
            this.accuracy.reset();
	    this.classifier.resetLearning();
	    this.createdOn=instancesSeen;
	    this.driftDetectionMethod=((ChangeDetector) getPreparedClassOption(this.driftOption)).copy();

	}
	
        public void trainOnInstance(Instance instance, double weight, long instancesSeen) {
	    this.instancesSeen=instancesSeen;

	    //Instance instance = (Instance) instance2.copy();
	    //System.out.println("[POISSON] weight:"+weight);
	    //weightedInstance.setWeight(instance.weight() * weight);
	    //this.classifier.trainOnInstance(weightedInstance);
	    //weightedInstance.setWeight(instance.weight() * weight);

	    double original_weight=instance.weight();

	    instance.setWeight(original_weight * weight);
	    this.classifier.trainOnInstance(instance);

	    instance.setWeight(original_weight);
	    
	    boolean correctlyClassifies=this.classifier.correctlyClassifies(instance);
	    this.driftDetectionMethod.input(correctlyClassifies ? 0 : 1);

	    // Check if there was a change
	    if(this.driftDetectionMethod.getChange()) {
		this.lastDriftOn = instancesSeen;
		this.reset();
	    }
	    
		    
        }

	
	public double[] getVotesForInstance(Instance instance) {
	    return this.classifier.getVotesForInstance(instance);

	    /*
	    DoubleVector vote=new DoubleVector(this.classifier
					       .getVotesForInstance(instance));
	    
            return vote.getArrayRef();
	    */
	}
	
	
    }

    
    /////////////////////////////////////////////////////////////////////////////////////
    //
    //// Learner Allocator
    //
    /////////////////////////////////////////////////////////////////////////////////////

    protected class learnerAllocator implements Serializable {

	private static final long serialVersionUID = 1L;
	
	protected ClassOption m_learnerClassOption;
	protected ClassOption m_driftDetectionMethod;
	//protected ClassOption m_warningDetectionMethod;

	protected int m_subspace;
	protected BasicClassificationPerformanceEvaluator m_classificationEvaluator;

	protected ARFHoeffdingTree m_baseLearner;


	public learnerAllocator(ClassOption learnerOption,
				ClassOption driftDetectionMethod){
	    //ClassOption warningDetectionMethod){

	    
	    this.m_learnerClassOption=learnerOption;
	    //this.m_subspace=subspaceSize;
  
	    this.m_driftDetectionMethod = driftDetectionMethod;
	    //this.m_warningDetectionMethod = warningDetectionMethod;

	    
	    this.m_baseLearner = (ARFHoeffdingTree)getPreparedClassOption(this.m_learnerClassOption);
	    this.m_classificationEvaluator = new BasicClassificationPerformanceEvaluator();
	    
	}

	public void setSubSpaceSize(int size){
	    this.m_baseLearner.subspaceSizeOption.setValue(size);
	}

	
	public ElasticBaseLearner alloc(int id, long instancesSeen){

	    return new ElasticBaseLearner(id, 
					  (ARFHoeffdingTree)this.m_baseLearner.copy(),
					  (BasicClassificationPerformanceEvaluator)
		                                  this.m_classificationEvaluator.copy(), 
					  instancesSeen, 
					  false, // disable BKG learner
					  true, // use drift
					  this.m_driftDetectionMethod,
					  //this.m_warningDetectionMethod,
					  false);
	    
	}


	
	/*
	ARFHoeffdingTree baseLearner=(ARFHoeffdingTree)getPreparedClassOption(this.treeLearnerOption);

	
	this.m_learnerAllocator = new ARFBaseLearnerAllocator(
				      this.disableBackgroundLearnerOption.isSet(),
				      this.disableDriftDetectionOption.isSet(),
				      baseLearner,
				      driftDetectionMethodOption,
				      warningDetectionMethodOption);

	 */
	
    }

    /////////////////////////////////////////////////////////////////////////////////////
    //
    //// OBJECT TO OBTAIN METRIC
    //
    /////////////////////////////////////////////////////////////////////////////////////

    
    protected static class Metric{
	static public double getMetric(ElasticBaseLearner l){
	    return 0;
	}
    }

    protected static class MetricAccuracy extends Metric{
	//@Override
	static public double getMetric(ElasticBaseLearner l){
	    return l.accuracy.get();
	}

    }
    
    
    /////////////////////////////////////////////////////////////////////////////////////
    //
    //// Group ENSEMBLE
    //
    /////////////////////////////////////////////////////////////////////////////////////

    /*
    class GroupDescriptor{

	protected int m_maxSize;
	protected int m_minSize;
	protected int m_head;

	//data?
	ElasticBaseLearner[] m_leaners; 
	protected learnerAllocator m_allocator;
	
	public GroupDescriptor(learnerAllocator alloc,
			       int maxSize,
			       int minSize,
			       int initialSize){
	    this.m_allocator=alloc;

	    this.m_maxSize=maxSize;
	    this.m_minSize=minSize;
	    
	    int growFactor=Math.max(minSize,initialSize);
	    grow(Group,growFactor);
	}
	
	public void grow(int resizeFactor,int instancesSeen){
	    if (isFull())
		return;

	    
	    for(int i=0; i<resizeFactor; i++){
		this.m_leaners[this.m_head]=this.m_allocator.alloc(this.m_head,
								   instancesSeen);
	    }

	    
	    this.m_groupCurrentElements[Group]+=factor;
	    this.m_currentSize+=factor;
	    System.out.println("GROW> Group size:" + this.m_groupCurrentElements[Group]
			       + " | factor:"+factor
			       + " | current size:"+this.size() );
	}
    };
    
    */


    
    class GroupEnsemble{

	//private static final long serialVersionUID = 1L;

	
	protected int m_maxGroups;
	protected int m_maxSize;
	protected int m_currentSize;
	protected ElasticBaseLearner[][] m_group;
	protected int[] m_groupMaxElements;
	protected int[] m_groupMinElements;
	protected int[] m_groupCurrentElements;

	protected learnerAllocator m_learnerAllocator;
	
	public GroupEnsemble(int MaxGroups, int maxElements,
			     learnerAllocator allocator){

	    this.m_maxGroups=MaxGroups;
	    this.m_maxSize=maxElements;
	    this.m_learnerAllocator = allocator;
	    
	    m_group = new ElasticBaseLearner[this.m_maxGroups][];
	    m_groupMaxElements = new int[this.m_maxGroups];
	    m_groupMinElements = new int[this.m_maxGroups];
	    m_groupCurrentElements = new int[this.m_maxGroups];
	    this.m_currentSize=0;
	}

	public int size(){
	    return this.m_currentSize;
	}

	
	public boolean isFull(){
	    return (this.size() >= this.m_maxSize);
	}
	
	public void initGroup(int Group,int initialSize,
			      int minSize,int maxSize){
	    if (Group > this.m_maxGroups)
		return;

	    
	    m_groupMaxElements[Group]=Math.min(maxSize,this.m_maxSize);
	    m_groupMinElements[Group]=minSize;
	    m_groupCurrentElements[Group]=0;

	    m_group[Group] = new ElasticBaseLearner[m_groupMaxElements[Group]];

	    int growFactor=Math.max(minSize,initialSize);
	    grow(Group,growFactor);
	}

	public int groupSize(int Group){
	    return m_groupCurrentElements[Group];
	}

	public ElasticBaseLearner learner(int Group,int idx){
	    return m_group[Group][idx];
	}

	
	
	
	public void groupInfo(int Group){
	    System.out.println("[ENSEMBLE " + Group + "]"
			       + " Size:" + this.m_groupCurrentElements[Group]
			       + " | max size:" + this.m_groupMaxElements[Group]
			       + " | min size:" + this.m_groupMinElements[Group]
			       );
	}

	public int grow(int Group, int resizeFactor){
	    if (isFull()){
		return -1;
	    }
	    
	    int factor1=this.m_maxSize-this.m_currentSize;
	    int factor2=this.m_groupMaxElements[Group]-this.m_groupCurrentElements[Group];
	    int factor=Math.min(factor1,Math.min(factor2,resizeFactor));

	    //groupInfo(Group);

	    int s=this.m_groupCurrentElements[Group];
	    int e=s+factor;
	    for(int i=s;i<e;i++){
		this.m_group[Group][i]=this.m_learnerAllocator.alloc(i,0);
	    }
	    this.m_groupCurrentElements[Group]=e;

	    /*
	    for(int i=0;i<factor;i++){
		int id=this.m_groupCurrentElements[Group];
		this.m_group[Group][id]=this.m_learnerAllocator.alloc(id,0);
		this.m_groupCurrentElements[Group]++;
	    }
	    */
	    this.m_currentSize+=factor;

	    return e;
	}

	
	public void shrink(int Group, int resizeFactor){
	    if (size() <=0){
		return;
	    }

	    int factor2=this.m_groupCurrentElements[Group] -this.m_groupMinElements[Group];
	    int factor=Math.min(factor2,resizeFactor);

	    /*
	    int e=this.m_groupCurrentElements[Group];
	    int s=e-factor;
	    for(int i=s;i<e;i++){
		this.m_group[Group][i]=nullptr
	    }
	    */
	    this.m_groupCurrentElements[Group]-=factor;
	    this.m_currentSize-=factor;


	    /*
	    System.out.println("SHRINK> group size:" + this.m_groupCurrentElements[Group]
			       + " | factor:"+factor
			       + " | current size:"+this.size() );
	    */
	}

	
	public void dumpGroup(int Group){
	    DecimalFormat nf4 = new DecimalFormat("#.0000");
	    
	    int s=groupSize(Group);
	    System.out.print("[GROUP:"+Group+"] ");
	    for(int i=0;i<s;i++){
		double acc=this.m_group[Group][i].accuracy.get();
		System.out.print(nf4.format(acc) + ", ");
	    }
	    System.out.println("");
	}


	public int findMin2(int Group){
	    return this.<MetricAccuracy>findMin3(Group);
	}

	//public <T extends Metric> int findMin3(int Group, Class<T> metric_type){
	//public <T extends Metric> int findMin3(int Group, T metric_type){
	public <T extends Metric> int findMin3(int Group){

	    int s=groupSize(Group);

	    double minValue=T.getMetric(this.m_group[Group][0]);
	    //double minValue=metric_type.getMetric(this.m_group[Group][0]);
	    //double minValue=Clazz<T>.getMetric(this.m_group[Group][0]);
	    int minValueIdx=0;
	    
	    for(int i=1;i<s;i++){
		double v=this.m_group[Group][i].accuracy.get();
		
		if (v<minValue){
		    minValue=v;
		    minValueIdx=i;
		}
	    }
	    return minValueIdx;
    
	}

	/*
	public int findMin(int Group,Metric m){
	    int s=groupSize(Group);
	    
	    double minValue=m.getMetric(this.m_group[Group][0]);
	    int minValueIdx=0;
	    
	    for(int i=1;i<s;i++){
		double v=this.m_group[Group][i].accuracy.get();
		
		if (v<minValue){
		    minValue=v;
		    minValueIdx=i;
		}
	    }
	    return minValueIdx;
	}
	*/
	
	// TODO findMin/Max should get metric obj, function
	// as a template or param
	public int findMin(int Group){
	    int s=groupSize(Group);
	    
	    double minValue=this.m_group[Group][0].accuracy.get();
	    int minValueIdx=0;
	    
	    for(int i=1;i<s;i++){
		double v=this.m_group[Group][i].accuracy.get();
		
		if (v<minValue){
		    minValue=v;
		    minValueIdx=i;
		}
	    }
	    return minValueIdx;
	}

	public int findMax(int Group){
	    int s=groupSize(Group);

	    double maxValue=this.m_group[Group][0].accuracy.get();
	    int maxValueIdx=0;
		
    	    for(int i=1;i<s;i++){
		
		double v=this.m_group[Group][i].accuracy.get();
		if (v>maxValue){
		    maxValue=v;
		    maxValueIdx=i;
		}
		
	    }
	    return maxValueIdx;
	}

	public int findMaxSkip(int Group){
	    int s=groupSize(Group);

	    double maxValue=this.m_group[Group][0].accuracy.get();
	    int maxValueIdx=0;
		
    	    for(int i=1;i<s;i++){

		if (this.m_group[Group][i].instancesSeen<20){
		    continue;
		}
		double v=this.m_group[Group][i].accuracy.get();
		if (v>maxValue){
		    maxValue=v;
		    maxValueIdx=i;
		}
		
	    }
	    return maxValueIdx;
	}


	
	
	/*
	 * Finds min  and move it to the end
	 * Return the other index
	 */
	public int findMoveMin(int Group){

	    int minValueIdx = findMin(Group);
	    
	    int s=groupSize(Group)-1;
	    this.swap(Group,minValueIdx,
		      Group,s);

	    return minValueIdx;
	}

	
	/*
	 * Find max and move it to pos 0
	 * Return the other index
	 */
	public int findMoveMax(int Group){

	    int maxValueIdx=findMax(Group);
	    this.swap(Group,maxValueIdx,
		      Group,0);
	    
	    return maxValueIdx;
	}

	
	public void swap(int Group1, int Idx1, int Group2, int Idx2){
	    ElasticBaseLearner l=this.m_group[Group1][Idx1];
	    this.m_group[Group1][Idx1]=this.m_group[Group2][Idx2];
	    this.m_group[Group2][Idx2]=l;
	}




	
	
    }


    /////////////////////////////////////////////////////////////////////////////////////
    //
    //// ELASTIC ENSEMBLE POLICY
    //
    /////////////////////////////////////////////////////////////////////////////////////

    protected interface Elastic{

	// y: target label
	// ys: shrunk ensemble prediction
	// yd: default ensemble prediction
	// yd: grown ensemble prediction
	public void addResults(int y, int ys, int yd, int yg);

	// returns:
	// 0          if noop
	// positive   should grow
	// negative   should shrink
	public int shouldResize();

	public void reset();

	public void grow();
	public void shrink();

	
    }

    
    protected class McnemarElastic implements Elastic{
	

	//private static final long serialVersionUID = 1L;

	protected ContingencyTable m_shrink;
	
	protected ContingencyTable m_grow;

	protected BasicAccuracy m_saccuracy;
	protected BasicAccuracy m_daccuracy;
	protected BasicAccuracy m_gaccuracy;
	
	public McnemarElastic(){
	    this.m_shrink=new ContingencyTable();
	    this.m_grow=new ContingencyTable();
	    
	    this.m_saccuracy = new BasicAccuracy();
	    this.m_daccuracy = new BasicAccuracy();
	    this.m_gaccuracy = new BasicAccuracy();
	    
	    reset();
	}
	
	@Override
	public void reset(){
	    this.m_shrink.reset();
	    this.m_grow.reset();

	    this.m_saccuracy.reset();
	    this.m_daccuracy.reset();
	    this.m_gaccuracy.reset();
	}

	@Override
	public void grow(){
	    reset();
	    //this.m_grow.reset();
	    //this.m_gaccuracy.set(this.m_daccuracy);
	    //this.m_gaccuracy.reset();
	    
	}

	@Override
	public void shrink(){
	    reset();

	    //this.m_shrink.reset();
	    //this.m_saccuracy.set(this.m_daccuracy);
	    //this.m_saccuracy.reset();

	    
	}

	
	@Override
	public void addResults(int y, int ys, int yd, int yg){
	    /*
	    System.out.println("[McNemarElastic] "
			       + " | y:"+y
			       + " | ys:"+ys
			       + " | yd:"+yd
			       + " | yg:"+yg
			       );
	    */
	    
	    this.m_shrink.addResult(y,ys,yd);
	    this.m_grow.addResult(y,yg,yd);
	    
	    this.m_saccuracy.addResult(y,ys);
	    this.m_daccuracy.addResult(y,yd);
	    this.m_gaccuracy.addResult(y,yg);
	}

	// returns:
	// 0          if noop
	// positive   should grow
	// negative   should shrink
	@Override
	public int shouldResize(){

	    double gtest=this.m_grow.mcnemar();
	    double stest=this.m_shrink.mcnemar();

	    double sacc= this.m_saccuracy.get();
    	    double dacc= this.m_daccuracy.get();
	    double gacc= this.m_gaccuracy.get();

	    /*
	    System.out.println("[SHOULDRESIZE]"
			       + " Mc_grow:"+gtest
			       + " Mc_shrink:"+stest
			       + " |"
			       + " ShrinkAcc:"+sacc
			       + " DefatulAcc:"+dacc
			       + " GrowAcc:"+gacc
			       );
			                           
	    */
	    //double sdelta=this.m_shrink.accuracyControl()-this.m_shrink.accuracyCase();
	    double sdelta=sacc - dacc;
	    //double gdelta=this.m_grow.accuracyControl()-this.m_grow.accuracyCase();
	    double gdelta=gacc - dacc;

	    /*
	    if (sdelta>0.00001){
		System.out.println("  =>SHRINK");
		return -1;
	    }

	    
	    if (gdelta > 0.0001){
		System.out.println("  =>GROW");
		return 1;
	    }

	    return 0;
	    */

	    /* 
	    // OPT-1
	    // parece qe tira bien
	    if (sdelta>0.00001){
		System.out.println("  SHOULDRESIZE:SHRINK");
		return -1;
	    }

	    
	    if (gdelta > 0.0001){
		//if(stest<2.71){
		if(stest<3.85){  //<- no va mal, parece
			//probando esto
			//if(stest<2.0){
		    System.out.println("  SHOULDRESIZE:GROW");
		    return 1;
		}
	    }

	    return 0;
	    */

	    /* 
	    // OPT-2
	    // Este esquema se acerca al ARF 
	    // agr_a => 89.76930
	    // covt =>  92.13523
	    // airl =>  66.418871 (19,42,85,16)
	    // elec =>  88.767876 (23,19,18,10)
	    if (sdelta>0.0){
		System.out.println("  SHOULDRESIZE:SHRINK");
		return -1;
	    }

	    if (gdelta > 0.0001){
		//if(stest<2.71){
		if(stest<3.85){  //<- no va mal, parece
		    //probando esto
		    //if(stest<2.0){
		    System.out.println("  SHOULDRESIZE:GROW");
		    return 1;
		}
	    }

	    return 0;
	    */

	    /*
	    // Mas lento que OPT-2
	    // agr_a => 89.61659 (10-11 leaners) 
	    // covt =>  92.13441374704826 (parece que usa menos learners que OPT-2)
	    // airl =>  66.55345088740282 (89,38,59,60)
            // elec =>  88.72417902542372 (17,19,18,10)
	    if (sdelta>0.0){
		System.out.println("  SHOULDRESIZE:SHRINK");
		return -1;
	    }

	    
	    if (gdelta > 0.0001){
		return 1;
	    }
	    // Al reves, user el propio grow como freno?
	    // que sentido tiene usar el stest como freno, y no pej el sdelta?
	    
	    return 0;
	    */

	    // Al reves, user el propio grow como freno?
	    // que sentido tiene usar el stest como freno, y no pej el sdelta?
	    
	    if (sdelta>0.0){
		//System.out.println("  SHOULDRESIZE:SHRINK");
		return -1;
	    }

	    if (gdelta > 0.0001){
		//if(stest<2.71){
		if(stest<3.85){  //<- no va mal, parece
		    //probando esto
		    //if(stest<2.0){
		    //System.out.println("  SHOULDRESIZE:GROW");
		    return 1;
		}
	    }

	    return 0;

	}

	
	
    }
    
    /////////////////////////////////////////////////////////////////////////////////////
    //
    //// ELASTIC POLICY
    //
    /////////////////////////////////////////////////////////////////////////////////////
    //protected interface SwapPolicy{
    //};
    
        

    
    /////////////////////////////////////////////////////////////////////////////////////
    //
    //// SWAP ENSEMBLE POLICY
    //
    /////////////////////////////////////////////////////////////////////////////////////
    //protected interface SwapPolicy{
    //};

    /*
    protected class AccuracyMetric{
	static public double get(ElasticBaseLearner learner){
	    return learner.accuracy.get();
	}
    }
    */

    
    protected class AccuracySwapPolicy{ // implements SwapPolocy{

	//private static final long serialVersionUID = 1L;

	
	//public AccuracySwapPolicy(){}
	
	public double metric(ElasticBaseLearner learner){
	    return learner.accuracy.get();
	}
		
	
	
	// swaps from frontGroup to backGroup
	// using metric function
	//public void swapAll(GroupEnsemble en,int frontGroupNum, int backGroup){
	public int swapAll(GroupEnsemble en,int frontGroupNum, int backGroupNum){
	    
	    // TODO: probably there is another way to swap all
	    // maybe linear?
	    int swapCount=0;
	    while (swap(en,frontGroupNum,backGroupNum)==true)
	    {
		swapCount++;
	    }
	    return swapCount;
	}


	// swaps from min frontGroup with max from backGroup
	// using metric function
	// returns true if items were swapped
	//         false otherwise
	public boolean  swap(GroupEnsemble en,int frontGroupNum, int backGroupNum){
	    //int frMinIdx= en.findMin2(frontGroupNum,AccuracyMetric.class);
	    int frMinIdx= en.findMin(frontGroupNum);
	    //int bkMaxIdx= en.findMax(backGroupNum);
	    int bkMaxIdx= en.findMaxSkip(backGroupNum);

	    //double frontValue=metric(en.learner(frontGroupNum,frMinIdx));
	    //double backValue=metric(en.learner(backGroupNum,bkMaxIdx));

	    ElasticBaseLearner flearner=en.learner(frontGroupNum,frMinIdx);
	    ElasticBaseLearner blearner=en.learner(backGroupNum,bkMaxIdx);

	    double frontValue=metric(flearner);
	    double backValue=metric(blearner);

	    /*
	    if (blearner.instancesSeen<=40){
		return false;
	    }
	    */

	    
	    if(frontValue<backValue){
		en.swap(frontGroupNum,frMinIdx,
			backGroupNum,bkMaxIdx);
		return true;
	    }
	    
	    return false;
	}

    };

   
     /////////////////////////////////////////////////////////////////////////////////////
    //
    //// ELASTIC POLICY
    //
    /////////////////////////////////////////////////////////////////////////////////////

    protected interface ElasticPolicy{

	
	public void init(Instance instance);
	public double[] getVotesForInstance(Instance instance);
	public void trainOnInstanceImpl(Instance instance);
	public void reset();

    }


    protected class ElasticPolicySwapOnly implements ElasticPolicy{

	//private static final long serialVersionUID = 1L;

	//protected int s_GROUPS=2;
	protected int s_GFRONT=0;
	protected int s_GCANDIDATE=1;

	protected ElasticConfig m_config;
	protected learnerAllocator m_learnerAllocator;

	protected GroupEnsemble m_ensemble;
	protected AccuracySwapPolicy m_swap;
	
	protected Random m_random = new Random(1);
	protected long m_instancesSeen;
	
	ElasticPolicySwapOnly(ElasticConfig config,
			      learnerAllocator alloc){
	    this.m_config = config;
	    this.m_learnerAllocator = alloc;
	    this.m_instancesSeen=0;
	}
	

	protected void _combineVotes(DoubleVector combinedVote, DoubleVector vote, double acc){
	
	    if (vote.sumOfValues() > 0.0) {
		vote.normalize();

		
		if (acc>0){
		    for(int v = 0 ; v < vote.numValues() ; ++v) {
			vote.setValue(v, vote.getValue(v) * acc);
		    }
		}
		
		combinedVote.addValues(vote);
	    }
	}

	protected void _resetGroup(int Group){
	    int e=this.m_ensemble.groupSize(Group);
	    for(int i=0;i<e;i++){
		this.m_ensemble.learner(Group,i).reset();
	    }
	}

    
	protected double[] _trainLearner(Instance instance,
					 ElasticBaseLearner l){
	
	    int trueClass=(int)instance.classValue();
	    double[] vote = l.getVotesForInstance(instance);


	    l.accuracy.addResult(trueClass,
				 vote);

	
	    l.trainOnInstance(instance,
			      MiscUtils.poisson(this.m_config.getLambda(),
						this.m_random),
			      this.m_instancesSeen);
	
	    return vote;
	}
				 
	protected void _trainRange(Instance instance,
				   int Group,int startIdx, int endIdx,
				   DoubleVector combinedVote){


	    int s=startIdx;
	    int e=endIdx;
	    //System.out.print("[TRAINRANGE:"+Group+"] Combine | ");
	    
	    for(int i=s;i<e;i++){
		//System.out.print("[i:"+i+"] ");
    
		ElasticBaseLearner l=this.m_ensemble.learner(Group,i);
		DoubleVector vote = new DoubleVector(_trainLearner(instance,l));
		_combineVotes(combinedVote, vote, l.accuracy.get());
	    }
	    //System.out.println("");

	}

	protected void _trainRange(Instance instance,
				   int Group,int startIdx, int endIdx){
	    int s=startIdx;
	    int e=endIdx;
	    //System.out.print("[TRAINRANGE:"+Group+"] NoCombine | ");

	    for(int i=s;i<e;i++){
		//System.out.print("[i:"+i+"] ");

		ElasticBaseLearner l=this.m_ensemble.learner(Group,i);
		//DoubleVector vote = new DoubleVector(_trainLearner(instance,l));
		_trainLearner(instance,l);

	    }
	    //System.out.println("");

	}

    
	protected void _trainGroup(int Group,
				   Instance instance,
				   DoubleVector combinedVote){

	    int e=this.m_ensemble.groupSize(Group);
	    _trainRange(instance,
			Group,0,e,
			combinedVote);
	}
    
	protected void _trainGroup(int Group,
				   Instance instance){


	    int e=this.m_ensemble.groupSize(Group);
	    _trainRange(instance,
			Group,0,e);
	
	}

	protected void _reserveGroups(int num){
	    this.m_ensemble = new GroupEnsemble(num,
						this.m_config.getLearnersMaxSize(),
						this.m_learnerAllocator
						);
	}

	protected void _initSwap(){
	    // Init Front Learners
	    this.m_ensemble.initGroup(s_GFRONT,
				      this.m_config.getFrontSize(),
				      10,
				      this.m_config.getLearnersMaxSize()
				      );

	    // Init Candidate Learners
	    this.m_ensemble.initGroup(s_GCANDIDATE,
				      this.m_config.getCandidatesSize(),
				      10,
				      this.m_config.getLearnersMaxSize()
				      );

	    this.m_swap = new AccuracySwapPolicy();

	}

	protected void _doSwap(){
	    int swaps=0;
	    //swaps=this.m_swap.swapAll(this.m_ensemble,s_GFRONT,s_GCANDIDATE);
	    boolean ret=this.m_swap.swap(this.m_ensemble,s_GFRONT,s_GCANDIDATE);
	    //swaps=(ret==true) ? 1 : 0;
	    //System.out.println("[SWAP] count:"+swaps);
	}

	
	protected void _dumpGroupStats(int Group){
	    int minIdx= this.m_ensemble.findMin(Group);
	    int maxIdx= this.m_ensemble.findMax(Group);

	    double maxAcc=this.m_ensemble.learner(Group,maxIdx).accuracy.get();
	    double minAcc=this.m_ensemble.learner(Group,minIdx).accuracy.get();

    
	    int s=this.m_ensemble.groupSize(Group);
	    double accAccum=0;
	    for(int i=0;i<s;i++){
		ElasticBaseLearner l=this.m_ensemble.learner(Group,i);
		double acc=l.accuracy.get();
		accAccum+=acc;
	    }
	    double accMean=accAccum/s;
	    /*
	    System.out.println("[GROUP " + Group + "] "
			       + "mean:"+accMean
			       + " | max:"+maxAcc
			       + " min:"+minAcc
			       + " dist:"+(maxAcc-minAcc)
			       );
	    //+ " | max:"+maxAcc
	    */		       
			       
	    
	}
	
	@Override
	public void reset() {
	}
	
	
	@Override
	public void init(Instance instance) {
	    
	    _reserveGroups(2);
	    _initSwap();
	    
	}

	@Override
	public double[] getVotesForInstance(Instance instance) {

	    DoubleVector combinedVote = new DoubleVector();
	    
	    int s=this.m_ensemble.groupSize(s_GFRONT);
	    for(int i=0;i<s;i++){
		ElasticBaseLearner l=this.m_ensemble.learner(s_GFRONT,i);
		DoubleVector vote = new DoubleVector(l.getVotesForInstance(instance));
		this._combineVotes(combinedVote, vote, l.accuracy.get());
	    
	    }
	    
	    return combinedVote.getArrayRef();
	}

	@Override
	public void trainOnInstanceImpl(Instance instance) {
	    ++this.m_instancesSeen;

	    double weight = instance.weight();
	    if (weight == 0.0)
		return;

	    this.m_ensemble.findMoveMin(s_GFRONT);
	    this._trainGroup(s_GFRONT,instance);
	    this.m_ensemble.findMoveMin(s_GFRONT);

	    
	    this.m_ensemble.findMoveMax(s_GCANDIDATE);
	    this._trainGroup(s_GCANDIDATE,instance);
	    this.m_ensemble.findMoveMax(s_GCANDIDATE);

	    //this._doSwap();
	    boolean ret=this.m_swap.swap(this.m_ensemble,s_GFRONT,s_GCANDIDATE);

	    _dumpGroupStats(s_GFRONT);
	    _dumpGroupStats(s_GCANDIDATE);
	    
	}
	
    }


    //////////////
    //////////////
    //////////////
        
    // ELASTIC POLICY: GROW ONLY FRONT
    /////////////////////////////////////////////////////////////////////////////////////

    protected class ElasticPolicyGrowFront extends ElasticPolicySwapOnly{

	//private static final long serialVersionUID = 1L;

	/*
	protected int s_GROUPS=3;
	protected int s_GFRONT=0;
	protected int s_GCANDIDATE=1;
	*/
	protected int s_GGROW=2;

	protected Elastic m_elastic;

	
	ElasticPolicyGrowFront(ElasticConfig config,
			       learnerAllocator alloc){
	    super(config,alloc);
	}

	
	@Override
	public void init(Instance instance) {
	    _reserveGroups(3);
	    //_initSwapGroups();
	    _initSwap();

	    // Used for growing the ensemble only
	    this.m_ensemble.initGroup(s_GGROW,
				      1,
				      1,
				      1
				      );


	    this.m_elastic = new McnemarElastic();

	    
	}

	/*
	protected void _trainCombine(){
	}
	*/	
	@Override
	public void trainOnInstanceImpl(Instance instance) {
	    ++this.m_instancesSeen;

	    int trueClass=(int)instance.classValue();
	    double weight = instance.weight();
	    if (weight == 0.0)
		return;


	    DoubleVector combinedVote = new DoubleVector();
	    DoubleVector voteCmax = new DoubleVector();
	    DoubleVector voteGmax = new DoubleVector();


	    //this.m_ensemble.findMoveMin(s_GFRONT);
	    _trainGroup(s_GFRONT,instance,combinedVote);

	    int ys=Utils.maxIndex(combinedVote.getArrayRef()); //.maxIndex();
	    int yd=Utils.maxIndex(combinedVote.getArrayRef()); //.maxIndex();
	
	    //
	    // Get & store vote for Cmax learner at pos 0
	    //
	
	    this.m_ensemble.findMoveMax(s_GCANDIDATE);
	    _trainRange(instance,
			s_GCANDIDATE,0,1,
			voteCmax);
	    // Train the rest without storing votes
	    int sc=this.m_ensemble.groupSize(s_GCANDIDATE);
	    _trainRange(instance,
			s_GCANDIDATE,1,sc);
	    this.m_ensemble.findMoveMax(s_GCANDIDATE);
	

	    this.m_ensemble.findMoveMax(s_GGROW);
	    _trainRange(instance,
			s_GGROW,0,1,
			voteGmax);
	
	    // Train the rest without storing votes
	    int sg=this.m_ensemble.groupSize(s_GGROW);
	    _trainRange(instance,
			s_GGROW,1,sg);
	    this.m_ensemble.findMoveMax(s_GGROW);
	
	
	    double cmax=this.m_ensemble.learner(s_GCANDIDATE,0).accuracy.get();
	    double gmax=this.m_ensemble.learner(s_GGROW,0).accuracy.get();
	
	    //int group=s_GCANDIDATE;
	    if (gmax > cmax){
		_combineVotes(combinedVote, voteGmax,gmax);
		//group=s_GGROW;
	    }else{
		_combineVotes(combinedVote, voteCmax,cmax);
		//group=s_GCANDIDATE;
	    }
	
	    int yg=Utils.maxIndex(combinedVote.getArrayRef()); //.maxIndex();

		
	    this.m_elastic.addResults(trueClass,
				      ys,yd,yg);

	    int resizeOperation=this.m_elastic.shouldResize();

	    if (resizeOperation==s_ELASTIC_OPERATION_GROW){

		int new_size=this.m_ensemble.grow(s_GFRONT,1);
		if (new_size>=0){
		    this.m_ensemble.swap(s_GFRONT,new_size-1,
					 s_GGROW,0);
		    //_resetGroup(s_GGROW);
		    this.m_elastic.reset();
		    //this.m_elastic.grow();
		}
	    }

	    this._doSwap();

	    //this.m_ensemble.groupInfo(s_GFRONT);
	    //this.m_ensemble.groupInfo(s_GCANDIDATE);
	    
	    
	}
     }
    
    // ELASTIC POLICY: GROW/SHRINK
    /////////////////////////////////////////////////////////////////////////////////////

    protected class ElasticPolicy1F0C extends ElasticPolicySwapOnly{

	//private static final long serialVersionUID = 1L;

	/*
	protected int s_GROUPS=3;
	protected int s_GFRONT=0;
	protected int s_GCANDIDATE=1;
	*/
	protected int s_GGROW=2;

	protected Elastic m_elastic;

	ElasticPolicy1F0C(ElasticConfig config,
			  learnerAllocator alloc){
	    super(config,alloc);
	}

	
	@Override
	public void init(Instance instance) {
	    _reserveGroups(3);
	    //_initSwapGroups();
	    _initSwap();
	    /*
	    // Used for growing the ensemble only
	    this.m_ensemble.initGroup(s_GGROW,
				      2,
				      2,
				      2
				      );
	    */
	    
	    this.m_ensemble.initGroup(s_GGROW,
				      1,
				      1,
				      1
				      );
	    
	    
	    this.m_elastic = new McnemarElastic();

	    
	}

	
	/*
	protected void _trainCombine(){
	}
	*/	
	//@Override
	public void trainOnInstanceImpl2(Instance instance) {
	    ++this.m_instancesSeen;

	    int trueClass=(int)instance.classValue();
	    double weight = instance.weight();
	    if (weight == 0.0)
		return;

	    
	    DoubleVector combinedVote = new DoubleVector();
	    //DoubleVector voteCmax = new DoubleVector();
	    //DoubleVector voteGmax = new DoubleVector();
	    
	    _trainGroup(s_GFRONT,instance);
	    //_trainGroup(s_GFRONT,instance,combinedVote);
	    _trainGroup(s_GCANDIDATE,instance);
	    _trainGroup(s_GGROW,instance);

	    this.m_ensemble.findMoveMin(s_GFRONT);
	    this.m_ensemble.findMoveMax(s_GCANDIDATE);
	    this.m_ensemble.findMoveMax(s_GGROW);


	    int e=this.m_ensemble.groupSize(s_GFRONT);
	    for(int i=0;i<(e-1);i++){
		//System.out.print("[i:"+i+"] ");
		ElasticBaseLearner l=this.m_ensemble.learner(s_GFRONT,i);
		double[] v = l.getVotesForInstance(instance);
		DoubleVector vote = new DoubleVector(v);
		_combineVotes(combinedVote, vote, l.accuracy.get());
	    }

	    int ys=Utils.maxIndex(combinedVote.getArrayRef());
	    
	    
	    ElasticBaseLearner l=this.m_ensemble.learner(s_GFRONT,e-1);
	    double[] lv = l.getVotesForInstance(instance);
	    DoubleVector lastVote = new DoubleVector(lv);

	    _combineVotes(combinedVote, lastVote, l.accuracy.get());

	    int yd=Utils.maxIndex(combinedVote.getArrayRef());

	    double cmax=this.m_ensemble.learner(s_GCANDIDATE,0).accuracy.get();
	    double gmax=this.m_ensemble.learner(s_GGROW,0).accuracy.get();

	    //ElasticBaseLearner l;
	    if (cmax>gmax){
		l=this.m_ensemble.learner(s_GCANDIDATE,0);
	    }else{
		l=this.m_ensemble.learner(s_GGROW,0);
	    }
	    
	    double[] gVote = l.getVotesForInstance(instance);
	    DoubleVector growVote = new DoubleVector(gVote);
	    _combineVotes(combinedVote, growVote, l.accuracy.get());


	    int yg=Utils.maxIndex(combinedVote.getArrayRef());


	    this.m_elastic.addResults(trueClass,
				      ys,yd,yg);


	    int resizeOperation=this.m_elastic.shouldResize();
	    if (resizeOperation==s_ELASTIC_OPERATION_GROW){
		int new_size=this.m_ensemble.grow(s_GFRONT,1);
		if (new_size>=0){
		    this.m_ensemble.swap(s_GFRONT,new_size-1,
					 s_GGROW,0);
		    this.m_elastic.grow();
		}
	    }else if (resizeOperation==s_ELASTIC_OPERATION_SHRINK){
		this.m_ensemble.shrink(s_GFRONT,1);
		this.m_elastic.shrink();
	    }

	    this._doSwap();
	    //this.m_ensemble.groupInfo(s_GFRONT);

	    
	}
	
	@Override
	public void trainOnInstanceImpl(Instance instance) {
	    ++this.m_instancesSeen;

	    int trueClass=(int)instance.classValue();
	    double weight = instance.weight();
	    if (weight == 0.0)
		return;


	    DoubleVector combinedVote = new DoubleVector();
	    DoubleVector voteCmax = new DoubleVector();
	    DoubleVector voteGmax = new DoubleVector();


	    //this.m_ensemble.findMoveMin(s_GFRONT);
	    //_trainGroup(s_GFRONT,instance,combinedVote);

	    int s=this.m_ensemble.groupSize(s_GFRONT);
	    this.m_ensemble.findMoveMin(s_GFRONT);
	    _trainRange(instance,
			s_GFRONT,0,s-1,
			combinedVote);
	    //this.m_ensemble.findMoveMin(s_GFRONT);
	    	    
	    int ys=Utils.maxIndex(combinedVote.getArrayRef()); //.maxIndex();

	    _trainRange(instance,
			s_GFRONT,s-1,s,
			combinedVote);
	    int yd=Utils.maxIndex(combinedVote.getArrayRef()); //.maxIndex();
	    //this.m_ensemble.findMoveMin(s_GFRONT);

	    
	    //
	    // Get & store vote for Cmax learner at pos 0
	    //
	
	    this.m_ensemble.findMoveMax(s_GCANDIDATE);
	    _trainRange(instance,
			s_GCANDIDATE,0,1,
			voteCmax);
	    // Train the rest without storing votes
	    int sc=this.m_ensemble.groupSize(s_GCANDIDATE);
	    _trainRange(instance,
			s_GCANDIDATE,1,sc);
	    // este
	    this.m_ensemble.findMoveMax(s_GCANDIDATE);
	

	    this.m_ensemble.findMoveMax(s_GGROW);
	    _trainRange(instance,
			s_GGROW,0,1,
			voteGmax);
	
	    // Train the rest without storing votes
	    int sg=this.m_ensemble.groupSize(s_GGROW);
	    _trainRange(instance,
			s_GGROW,1,sg);
	    // este
	    this.m_ensemble.findMoveMax(s_GGROW);

	    
	
	    double cmax=this.m_ensemble.learner(s_GCANDIDATE,0).accuracy.get();
	    double gmax=this.m_ensemble.learner(s_GGROW,0).accuracy.get();
	
	    //int group=s_GCANDIDATE;
	    if (gmax > cmax){
		_combineVotes(combinedVote, voteGmax,gmax);
		//group=s_GGROW;
	    }else{
		_combineVotes(combinedVote, voteCmax,cmax);
		//group=s_GCANDIDATE;
	    }
	
	    int yg=Utils.maxIndex(combinedVote.getArrayRef()); //.maxIndex();

	    
		
	    this.m_elastic.addResults(trueClass,
				      ys,yd,yg);


	    	
	    int resizeOperation=this.m_elastic.shouldResize();
	    /*
	    int resizeOperation=0;
	    long trigger=this.m_instancesSeen%50;
	    if (trigger==0){
		resizeOperation=this.m_elastic.shouldResize();
	    }
	    */
	    if (resizeOperation==s_ELASTIC_OPERATION_GROW){
		int new_size=this.m_ensemble.grow(s_GFRONT,1);
		if (new_size>=0){
		    this.m_ensemble.swap(s_GFRONT,new_size-1,
					 s_GGROW,0);
		    //_resetGroup(s_GGROW);
		    //this.m_elastic.reset();
		    this.m_elastic.grow();
		    //System.out.println("  =>GROW");

		}
	    }else if (resizeOperation==s_ELASTIC_OPERATION_SHRINK){
		//int new_size=this.m_ensemble.shrink(s_GFRONT,1);
		this.m_ensemble.shrink(s_GFRONT,1);
		//_resetGroup(s_GGROW);
		this.m_elastic.shrink();
		//System.out.println("  =>SHRINK");

	    }

	    this._doSwap();

	    //this.m_ensemble.groupInfo(s_GFRONT);
	    //this.m_ensemble.groupInfo(s_GCANDIDATE);
	    
	    
	}
     }

    
    /////////////////////////////////////////////////////////////////////////////////////
    //
    //// ELASTIC RANDOM FOREST
    //
    /////////////////////////////////////////////////////////////////////////////////////
        
    private static final long serialVersionUID = 1L;
    
    private static final int s_frontOptDefSize=10;
    private static final int s_candOptDefSize=10;
    
    private static final int s_GROUPS=3;
    private static final int s_GFRONT=0;
    private static final int s_GCANDIDATE=1;
    private static final int s_GGROW=2;
    

    private static final int s_ELASTIC_OPERATION_SHRINK=-1;
    private static final int s_ELASTIC_OPERATION_KEEP=0;
    private static final int s_ELASTIC_OPERATION_GROW=1;
    
    
    //protected Random classifierRandom;

    
    // Configurations
    /////////////////////////////////////////////////////////////////////////////////////

    // -a
    public FloatOption lambdaOption = new FloatOption("lambda", 'a',
						      "The lambda parameter for bagging.",
						      6.0, 1.0, Float.MAX_VALUE);

    // -c
    public IntOption candidateSizeOption=new IntOption("candidateSize", 'c',
						       "The number of candidate learners.",
						       s_candOptDefSize,
						       1,
						       Integer.MAX_VALUE);
    

    // -f
    public IntOption frontSizeOption=new IntOption("frontSize", 'f',
						   "The number of front learners.",
						   s_frontOptDefSize,
						   1,
						   Integer.MAX_VALUE);

    
    // -l
    public ClassOption treeLearnerOption = new ClassOption("treeLearner", 'l',
							   "Random Forest Tree.",
							   ARFHoeffdingTree.class,
							   "ARFHoeffdingTree -e 2000000 -g 50 -c 0.01");

	/*
    // -p
    public ClassOption warningDetectionMethodOption = new ClassOption("warningDetectionMethod", 'p',
				        "Change detector for warnings (start training bkg learner)",
					ChangeDetector.class,
					"ADWINChangeDetector -a 1.0E-4");
	*/
    
    // -s
    public IntOption maxSizeOption = new IntOption("maxSize", 's',
						   "Maximum learners (front+candidate).",
						   100, 20, Integer.MAX_VALUE);

    
    // -x
    public ClassOption driftDetectionMethodOption = new ClassOption("driftDetectionMethod", 'x',
					        "Change detector for drifts and its parameters",
						ChangeDetector.class,
						"ADWINChangeDetector -a 1.0E-5");


    

    // M
    /////////////////////////////////////////////////////////////////////////////////////

    protected ElasticConfig m_config;
    
    //protected GroupEnsemble m_ensemble;
    protected learnerAllocator m_learnerAllocator;

    protected ElasticPolicy m_elasticPolicy;

    protected boolean m_init;
    
    //protected AccuracySwapPolicy m_swap;
    //protected Elastic m_elastic;

    //protected DoubleVector m_combinedVote;// = new DoubleVector();
    //protected DoubleVector m_combined1;// = new DoubleVector();
    //protected DoubleVector m_combined2; // = new DoubleVector();

    //public long instancesSeen;

    
    
    // Methods
    /////////////////////////////////////////////////////////////////////////////////////
    
    @Override
    public String getPurposeString() {
        return "Elastic Swap Random Forest algorithm for evolving data streams from Marron et al.";
    }


    @Override
    public void resetLearningImpl() {

	this.m_learnerAllocator = new learnerAllocator(this.treeLearnerOption,
						       this.driftDetectionMethodOption
						       );
	//this.warningDetectionMethodOption
	//					       );
	
	//System.out.println("resetLearningimpl");

	//this.classifierRandom=new Random(1);
	//this.instancesSeen=0;
	//System.out.println("[SWAPONLY] init");

	this.m_init=false;
    }
    
    /*
    protected void _combineVotes(DoubleVector combinedVote, DoubleVector vote){
	
	if (vote.sumOfValues() > 0.0) {
	    vote.normalize();
	    combinedVote.addValues(vote);
	}
    }
    */
    /*
    protected void _combineVotes(DoubleVector combinedVote, DoubleVector vote, double acc){
	
	if (vote.sumOfValues() > 0.0) {
	    vote.normalize();
	    
	    if (acc>0){
	    for(int v = 0 ; v < vote.numValues() ; ++v) {
		    vote.setValue(v, vote.getValue(v) * acc);
		}
	    }
	    
	    combinedVote.addValues(vote);
	}
    }


    
    protected void _resetGroup(int Group){

	int e=this.m_ensemble.groupSize(Group);
	for(int i=0;i<e;i++){
	    this.m_ensemble.learner(Group,i).reset();
	}

    }

    
    protected double[] _trainLearner(Instance instance,
				     ElasticBaseLearner l){
	
	int trueClass=(int)instance.classValue();
	double[] vote = l.getVotesForInstance(instance);

	l.accuracy.addResult(trueClass,
			     vote);

	
	l.trainOnInstance(instance,
			  MiscUtils.poisson(this.lambdaOption.getValue(),
					    this.classifierRandom),
			  this.instancesSeen);
	
	return vote;
    }
				 
    protected void _trainRange(Instance instance,
			       int Group,int startIdx, int endIdx,
			       DoubleVector combinedVote){


	int s=startIdx;
	int e=endIdx;
	System.out.print("[TRAINRANGE:"+Group+"] Combine | ");

	for(int i=s;i<e;i++){
	    System.out.print("[i:"+i+"] ");
    
	    ElasticBaseLearner l=this.m_ensemble.learner(Group,i);
	    DoubleVector vote = new DoubleVector(_trainLearner(instance,l));
	    _combineVotes(combinedVote, vote, l.accuracy.get());
	}
	System.out.println("");

    }

    protected void _trainRange(Instance instance,
			       int Group,int startIdx, int endIdx){
	int s=startIdx;
	int e=endIdx;
	System.out.print("[TRAINRANGE:"+Group+"] NoCombine | ");

	for(int i=s;i<e;i++){
	    System.out.print("[i:"+i+"] ");

	    ElasticBaseLearner l=this.m_ensemble.learner(Group,i);
	    //DoubleVector vote = new DoubleVector(_trainLearner(instance,l));
	    _trainLearner(instance,l);

	}
	System.out.println("");

    }

    
    protected void _trainGroup(int Group,
			       Instance instance,
			       DoubleVector combinedVote){

	int e=this.m_ensemble.groupSize(Group);
	_trainRange(instance,
		    Group,0,e,
		    combinedVote);
    }
    
    protected void _trainGroup(int Group,
			       Instance instance){


	int e=this.m_ensemble.groupSize(Group);
	_trainRange(instance,
		    Group,0,e);
	
    }
    */
    
    @Override
    public void trainOnInstanceImpl(Instance instance) {
	if (this.m_init==false){
	    _initEnsemble(instance);
	}
	this.m_elasticPolicy.trainOnInstanceImpl(instance);
    }
    
    /*
      TODO: ==> stop hardcoding things!! <====
      This implementation is garbage
      Need a propper elastic manager to decouple how ensemble is trained / resized
      Should be able also to selec metric
     */
    public void trainOnInstanceImpl2(Instance instance) {

	
        //++this.instancesSeen;
	//if (this.m_ensemble == null)
	//    _initEnsemble(instance);

	/*
	int trueClass=(int)instance.classValue();
	double weight = instance.weight();
	if (weight == 0.0)
	    return;

	DoubleVector combinedVote = new DoubleVector();
	DoubleVector voteCmax = new DoubleVector();
	DoubleVector voteGmax = new DoubleVector();


	//this.m_ensemble.findMoveMin(s_GFRONT);
	_trainGroup(s_GFRONT,instance,combinedVote);
	*/
	/*
	this.m_ensemble.findMoveMax(s_GCANDIDATE);
	_trainGroup(s_GCANDIDATE,instance);
	this.m_ensemble.findMoveMax(s_GCANDIDATE);
	*/
	
	/*
	int ys=Utils.maxIndex(combinedVote.getArrayRef()); //.maxIndex();
	int yd=Utils.maxIndex(combinedVote.getArrayRef()); //.maxIndex();
	
	//
	// Get & store vote for Cmax learner at pos 0
	//
	
	this.m_ensemble.findMoveMax(s_GCANDIDATE);
	_trainRange(instance,
		    s_GCANDIDATE,0,1,
		    voteCmax);
	// Train the rest without storing votes
	int sc=this.m_ensemble.groupSize(s_GCANDIDATE);
	_trainRange(instance,
		    s_GCANDIDATE,1,sc);
	this.m_ensemble.findMoveMax(s_GCANDIDATE);
	
	*/
	/*
	hay algo con esta parte..
	    si pongo el traingroup, el acc en AGR baja.
	    deberia ser equivalente no??

	*/

	/* trainRange No Combine Votes:  89.5247
	   traingroup                 :  89.2739
	   trainRange Combin Votes    :  89.5247

	   FIND:
	   traing group le falta el el findMoveMax despues.
	   pq esto afecta??
	   No es necesario si solo hay swap no????
	   (pq el swap busca el max)
	   probar pej: cambiar el findMoveMax el if value> por if value >=

	 */
	/*
	int sc=this.m_ensemble.groupSize(s_GCANDIDATE);
	_trainRange(instance,
		    s_GCANDIDATE,0,sc);
		    //voteCmax);
	this.m_ensemble.findMoveMax(s_GCANDIDATE);
	*/
	
	//
	// Same procedure with GROW group
	//
	/*
	this.m_ensemble.findMoveMax(s_GGROW);
	
	_trainRange(instance,
		    s_GGROW,0,1,
		    voteGmax);
	
	
	// Train the rest without storing votes
	int sg=this.m_ensemble.groupSize(s_GGROW);
	_trainRange(instance,
		    s_GGROW,1,sg);
	this.m_ensemble.findMoveMax(s_GGROW);
	
	
	double cmax=this.m_ensemble.learner(s_GCANDIDATE,0).accuracy.get();
	double gmax=this.m_ensemble.learner(s_GGROW,0).accuracy.get();
	
	//int group=s_GCANDIDATE;
	if (gmax > cmax){
	    _combineVotes(combinedVote, voteGmax);
	    //group=s_GGROW;
	}else{
	    _combineVotes(combinedVote, voteCmax);
	    //group=s_GCANDIDATE;
	}
	*/
	/*
	//int yg=combinedVote.maxIndex();
	int yg=Utils.maxIndex(combinedVote.getArrayRef()); //.maxIndex();

	
 
		
	this.m_elastic.addResults(trueClass,
				  ys,yd,yg);
	*/
	/*
	int resizeOperation=this.m_elastic.shouldResize();

	if (resizeOperation==s_ELASTIC_OPERATION_GROW){

	    int new_size=this.m_ensemble.grow(s_GFRONT,1);
	    if (new_size>=0){
		this.m_ensemble.swap(s_GFRONT,new_size-1,
				     s_GGROW,0);
		//_resetGroup(s_GGROW);
		this.m_elastic.reset();
		//this.m_elastic.grow();

	    }
	    
	}
	*/
	/*
	int swaps=0;
	//swaps=this.m_swap.swapAll(this.m_ensemble,s_GFRONT,s_GCANDIDATE);
	boolean ret=this.m_swap.swap(this.m_ensemble,s_GFRONT,s_GCANDIDATE);
	swaps=(ret==true) ? 1 : 0;
	System.out.println("[SWAP] count:"+swaps);
	*/
	//this.m_ensemble.findMoveMax(s_GCANDIDATE);
	//this.m_ensemble.findMoveMax(s_GGROW);

	// display group info 
	//this.m_ensemble.groupInfo(s_GFRONT);
	//this.m_ensemble.groupInfo(s_GCANDIDATE);

    }

   
 
    
    @Override
    public double[] getVotesForInstance(Instance instance) {

	if (this.m_init==false){
	    _initEnsemble(instance);
	}

	return this.m_elasticPolicy.getVotesForInstance(instance);
	
	/*
	if (this.m_ensemble == null)
	    _initEnsemble(instance);

	DoubleVector combinedVote = new DoubleVector();

	
	int s=this.m_ensemble.groupSize(s_GFRONT);
	for(int i=0;i<s;i++){
	    ElasticBaseLearner l=this.m_ensemble.learner(s_GFRONT,i);
	    DoubleVector vote = new DoubleVector(l.getVotesForInstance(instance));
	    _combineVotes(combinedVote, vote, l.accuracy.get());
	    
	}

	return combinedVote.getArrayRef();

	*/
	
    }

    
    @Override
    public boolean isRandomizable() {
        return true;
    }

    @Override
    public void getModelDescription(StringBuilder arg0, int arg1) {
    }

    @Override
    protected Measurement[] getModelMeasurementsImpl() {
        return null;
    }

    protected void _initEnsemble(Instance instance) {

	//System.out.println("[ELASTICRANDOMFORES] init");

	//this.subspaceSize = this.computeSubSpaceSize(instance);
        int n = instance.numAttributes()-1; // Ignore class label ( -1 )
	int subspaceSize = (int) Math.round(Math.sqrt(n)) + 1;
        if(subspaceSize < 0)
            subspaceSize = n + subspaceSize;
	
        if(subspaceSize <= 0)
            subspaceSize = 1;
        // m > n, then it should use n
        if(subspaceSize > n)
            subspaceSize = n;

	
	//System.out.println("subspace>"+subspaceSize);
	this.m_learnerAllocator.setSubSpaceSize(subspaceSize);

	this.m_config = new ElasticConfig(this.lambdaOption,
					  this.candidateSizeOption,
					  this.frontSizeOption,
					  this.maxSizeOption);


	
	this.m_elasticPolicy = new ElasticPolicySwapOnly(this.m_config,
							 this.m_learnerAllocator);


	
	/*
	this.m_elasticPolicy = new ElasticPolicyGrowFront(this.m_config,
	                                                  this.m_learnerAllocator);
	*/
	/*
	this.m_elasticPolicy = new ElasticPolicy1F0C(this.m_config,
						     this.m_learnerAllocator);

	*/
	this.m_elasticPolicy.init(instance);
	this.m_init=true;
	
	
	/*
	this.m_ensemble = new GroupEnsemble(s_GROUPS,
					    this.maxSizeOption.getValue(),
					    this.m_learnerAllocator
					    );
	// Init Front Learners
	this.m_ensemble.initGroup(s_GFRONT,
				  this.frontSizeOption.getValue(),
				  10,
				  this.maxSizeOption.getValue()
				  );
	// Init Candidate Learners
	this.m_ensemble.initGroup(s_GCANDIDATE,
				  this.candidateSizeOption.getValue(),
				  10,
				  this.maxSizeOption.getValue()
				  );
	
	// Used for growing the ensemble only
	this.m_ensemble.initGroup(s_GGROW,
				  1,
				  1,
				  1
				  );
	
	*/
	// TODO:
	// K

	//System.out.println("Lambda:" + this.lambdaOption.getValue());

	
	/*
	this.m_swap = new AccuracySwapPolicy();
	this.m_elastic = new McnemarElastic();

	this.m_combined1 = new DoubleVector();
	this.m_combined2 = new DoubleVector();
	this.m_combinedVote = new DoubleVector();
	*/
    }

    

}
