/*
 *    ID3.java
 *    Copyright (C) 2005 Liangxiao Jiang
 *
 */

package weka.classifiers.sk;

import weka.classifiers.*;
import weka.core.*;
import java.util.*;

/**
 * Implement ID3 classifier.
 */
public class Id3_LeafKNND2 extends Classifier
{

	/** The node's successors. */
	private Id3_LeafKNND2[] m_Successors;

	/** Attribute used for splitting. */
	private Attribute m_Attribute;

	/** The instances of the leaf node. */
	private Instances m_Instances;

	/**
	 * Builds ID3 decision tree classifier.
	 * 
	 * @param data
	 *            the training data
	 * @exception Exception
	 *                if classifier can't be built successfully
	 */
	public void buildClassifier(Instances data) throws Exception
	{

		// Build ID3 tree
		makeTree(data);
	}

	/**
	 * Method building ID3 tree using information gain measure
	 * 
	 * @param data
	 *            the training data
	 * @exception Exception
	 *                if decision tree can't be built successfully
	 */
	private void makeTree(Instances data) throws Exception
	{

		// Check if no instances have reached this node
		if (data.numInstances() == 0)
		{
			m_Attribute = null;
			m_Instances = new Instances(data);
			return;
		}
		// Compute attribute with maximum split value.
		double impurityReduce = 0;
		double maxValue = 0;
		int maxIndex = -1;
		for (int i = 0; i < data.numAttributes(); i++)
		{
			if (i == data.classIndex())
				continue;
			impurityReduce = computeEntropyReduce(data, data.attribute(i));
			if (impurityReduce > maxValue)
			{
				maxValue = impurityReduce;
				maxIndex = i;
			}
		}
		// Make leaf if information gain is zero, otherwise create successors.
		if (Utils.eq(maxValue, 0))
		{
			m_Attribute = null;
			m_Instances = new Instances(data);
			return;
		} else
		{
			m_Attribute = data.attribute(maxIndex);
			Instances[] splitData = splitData(data, m_Attribute);
			m_Successors = new Id3_LeafKNND2[m_Attribute.numValues()];
			for (int j = 0; j < m_Attribute.numValues(); j++)
			{
				m_Successors[j] = new Id3_LeafKNND2();
				m_Successors[j].makeTree(splitData[j]);
			}
		}
	}

	/**
	 * Splits a dataset according to the values of a nominal attribute.
	 * 
	 * @param data
	 *            the data which is to be split
	 * @param att
	 *            the attribute to be used for splitting
	 * @return the sets of instances produced by the split
	 */
	private Instances[] splitData(Instances data, Attribute att)
	{
		int numAttValues = att.numValues();
		Instances[] splitData = new Instances[numAttValues];
		for (int j = 0; j < numAttValues; j++)
		{
			splitData[j] = new Instances(data, 0);
		}
		int numInstances = data.numInstances();
		for (int i = 0; i < numInstances; i++)
		{
			int attVal = (int) data.instance(i).value(att);
			splitData[attVal].add(data.instance(i));
		}
		return splitData;
	}

	/**
	 * Computes information gain for an attribute.
	 * 
	 * @param data
	 *            the data for which info gain is to be computed
	 * @param att
	 *            the attribute
	 * @return the information gain for the given attribute and data
	 */
	private double computeEntropyReduce(Instances data, Attribute att)
			throws Exception
	{

		double entropyReduce = computeEntropy(data);
		Instances[] splitData = splitData(data, att);
		for (int j = 0; j < att.numValues(); j++)
		{
			if (splitData[j].numInstances() > 0)
			{
				entropyReduce -= ((double) splitData[j].numInstances() / (double) data
						.numInstances()) * computeEntropy(splitData[j]);
			}
		}
		return entropyReduce;
	}

	/**
	 * Computes the entropy of a dataset.
	 * 
	 * @param data
	 *            the data for which entropy is to be computed
	 * @return the entropy of the data's class distribution
	 */
	private double computeEntropy(Instances data) throws Exception
	{

		int numClasses = data.numClasses();
		int numInstances = data.numInstances();
		double[] classCounts = new double[numClasses];
		for (int i = 0; i < numInstances; i++)
		{
			int classVal = (int) data.instance(i).classValue();
			classCounts[classVal]++;
		}
		for (int i = 0; i < numClasses; i++)
		{
			classCounts[i] /= numInstances;
		}
		double Entropy = 0;
		for (int i = 0; i < numClasses; i++)
		{
			Entropy -= classCounts[i] * log2(classCounts[i], 1);
		}
		return Entropy;
	}

	/**
	 * compute the logarithm whose base is 2.
	 * 
	 * @param args
	 *            x,y are numerator and denominator of the fraction.
	 * @return the natual logarithm of this fraction.
	 */
	private double log2(double x, double y)
	{

		if (x < 1e-6 || y < 1e-6)
			return 0.0;
		else
			return Math.log(x / y) / Math.log(2);
	}

	/**
	 * Computes class distribution for instance using decision tree.
	 * 
	 * @param instance
	 *            the instance for which distribution is to be computed
	 * @return the class distribution for the given instance
	 */
	public double[] distributionForInstance(Instance instance) throws Exception
	{
		if (m_Attribute == null)
		{
			KNN_WeightD2 weightD2=new KNN_WeightD2();
			weightD2.buildClassifier(m_Instances);
			return weightD2.distributionForInstance(instance);
		} 
		else
		{
			return m_Successors[(int) instance.value(m_Attribute)]
					.distributionForInstance(instance);
		}
	}

	/**
	 * Compute the distribution.
	 * 
	 * @param data
	 *            the training data
	 * @exception Exception
	 *                if classifier can't be built successfully
	 */
	private double[] computeDistribution(Instances data) throws Exception
	{

		int numClasses = data.numClasses();
		double[] probs = new double[numClasses];
		double[] classCounts = new double[numClasses];
		int numInstances = data.numInstances();
		for (int i = 0; i < numInstances; i++)
		{
			int classVal = (int) data.instance(i).classValue();
			classCounts[classVal]++;
		}
		for (int i = 0; i < numClasses; i++)
		{
			probs[i] = (classCounts[i] + 1.0) / (numInstances + numClasses);
		}
		Utils.normalize(probs);
		return probs;
	}

	/**
	 * Main method.
	 * 
	 * @param args
	 *            the options for the classifier
	 */
	public static void main(String[] args)
	{

		try
		{
			System.out.println(Evaluation.evaluateModel(new Id3_LeafKNND2(), args));
		} catch (Exception e)
		{
			System.err.println(e.getMessage());
		}
	}

}
