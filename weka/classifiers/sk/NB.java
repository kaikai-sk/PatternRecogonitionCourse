package weka.classifiers.sk;

import weka.core.*;
import weka.classifiers.*;

/**
 * Implement the NB classifier.
 */
public class NB extends Classifier
{
	/** The number of class and each attribute value occurs in the dataset 
	 * 数据集中每个属性取值出现的个数     和    分类的个数    P(xk|ci)*/
	private double[][] m_ClassAttCounts;

	/** The number of each class value occurs in the dataset 
	 * 数据集中每个分类属性取值出现的个数*/
	private double[] m_ClassCounts;

	/** The number of values for each attribute in the dataset 
	 * 数据集中每个属性取值的个数*/
	private int[] m_NumAttValues;

	/** The starting index of each attribute in the dataset 
	 * 数据集中每个属性的开始的索引*/
	private int[] m_StartAttIndex;

	/** The number of values for all attributes in the dataset 
	 * 数据集中所有属性的取值的个数*/
	private int m_TotalAttValues;

	/**分类属性取值的个数 */
	private int m_NumClasses;

	/** 数据集中所有的属性的个数（包括分类属性） */
	private int m_NumAttributes;

	/** 训练样本的个数*/
	private int m_NumInstances;

	/** 分类属性的索引 */
	private int m_ClassIndex;

	public NB(Instances data)
	{
		initVariables(data);
	}
	
	public NB()
	{
		
	}
	/**
	 * Generates the classifier.
	 * 
	 * @param instances
	 *            set of instances serving as training data
	 * @exception Exception
	 *                if the classifier has not been generated successfully
	 */
	public void buildClassifier(Instances instances) throws Exception
	{
		initVariables(instances);
	}
	
	/**
	 * 初始化所有的成员变量
	 * @param instances
	 */
	private void initVariables(Instances instances)
	{
		// reset variable
		m_NumClasses = instances.numClasses();
		m_ClassIndex = instances.classIndex();
		m_NumAttributes = instances.numAttributes();
		m_NumInstances = instances.numInstances();
		m_TotalAttValues = 0;
		// allocate space for attribute reference arrays
		//对于属性引用数组分配空间
		m_StartAttIndex = new int[m_NumAttributes];
		m_NumAttValues = new int[m_NumAttributes];
		// set the starting index of each attribute and the number of values for
		// each attribute and the total number of values for all attributes(not
		// including class).
		for (int i = 0; i < m_NumAttributes; i++)
		{
			if (i != m_ClassIndex)
			{
				m_StartAttIndex[i] = m_TotalAttValues;
				m_NumAttValues[i] = instances.attribute(i).numValues();
				m_TotalAttValues += m_NumAttValues[i];
			} 
			else
			{	//保存分类属性的相关变量
				m_StartAttIndex[i] = -1;
				m_NumAttValues[i] = m_NumClasses;
			}
		}
		// allocate space for counts and frequencies
		m_ClassCounts = new double[m_NumClasses];
		/*		sunny overcast rain......weak strong
		 * yes
		 * no
		 */
		m_ClassAttCounts = new double[m_NumClasses][m_TotalAttValues];
		// Calculate the counts
		for (int k = 0; k < m_NumInstances; k++)
		{
			int classVal = (int) instances.instance(k).classValue();
			m_ClassCounts[classVal]++;
			int[] attIndex = new int[m_NumAttributes];
			for (int i = 0; i < m_NumAttributes; i++)
			{
				if (i == m_ClassIndex)
				{
					attIndex[i] = -1;
				}
				else
				{
					attIndex[i] = m_StartAttIndex[i]
							+ (int) instances.instance(k).value(i);
					m_ClassAttCounts[classVal][attIndex[i]]++;
				}
			}
		}
	}

	/**
	 * Calculates the class membership probabilities for the given test instance
	 * 
	 * @param instance
	 *            the instance to be classified
	 * @return predicted class probability distribution
	 * @exception Exception
	 *                if there is a problem generating the prediction
	 */
	public double[] distributionForInstance(Instance instance) throws Exception
	{
		// Definition of local variables
		double[] probs = new double[m_NumClasses];
		// store instance's att values in an int array
		int[] attIndex = new int[m_NumAttributes];
		for (int att = 0; att < m_NumAttributes; att++)
		{
			if (att == m_ClassIndex)
				attIndex[att] = -1;
			else
				attIndex[att] = m_StartAttIndex[att]
						+ (int) instance.value(att);
		}
		// calculate probabilities for each possible class value
		for (int classVal = 0; classVal < m_NumClasses; classVal++)
		{
			probs[classVal] = (m_ClassCounts[classVal] + 1.0)
					/ (m_NumInstances + m_NumClasses);
			for (int att = 0; att < m_NumAttributes; att++)
			{
				if (attIndex[att] == -1)
					continue;
				probs[classVal] *= (m_ClassAttCounts[classVal][attIndex[att]] + 1.0)
						/ (m_ClassCounts[classVal] + m_NumAttValues[att]);
			}
		}
		Utils.normalize(probs);
		return probs;
	}

	/**
	 * Main method for testing this class.
	 * 
	 * @param argv
	 *            the options
	 */
	public static void main(String[] argv)
	{
		try
		{
			System.out.println(Evaluation.evaluateModel(new NB(), argv));
		} 
		catch (Exception e)
		{
			e.printStackTrace();
			System.err.println(e.getMessage());
		}
	}

}
