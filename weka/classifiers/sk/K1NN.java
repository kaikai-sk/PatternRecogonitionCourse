/*
 *    This program is free software; you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation; either version 2 of the License, or
 *    (at your option) any later version.
 *
 *    This program is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public License
 *    along with this program; if not, write to the Free Software
 *    Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 */

/*
 *    IB1.java
 *    Copyright (C) 1999 University of Waikato, Hamilton, New Zealand
 *
 */

package weka.classifiers.sk;

import weka.classifiers.Classifier;
import weka.classifiers.UpdateableClassifier;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.RevisionUtils;
import weka.core.TechnicalInformation;
import weka.core.TechnicalInformationHandler;
import weka.core.Utils;
import weka.core.Capabilities.Capability;
import weka.core.TechnicalInformation.Field;
import weka.core.TechnicalInformation.Type;

import java.util.Enumeration;

public class K1NN extends Classifier 
{

	/** The training instances used for classification. */
	private Instances m_Train;

	/** The minimum values for numeric attributes. */
	private double[] m_MinArray;

	/** The maximum values for numeric attributes. */
	private double[] m_MaxArray;

	/**
	 * Returns default capabilities of the classifier.
	 * @return the capabilities of this classifier
	 */
	public Capabilities getCapabilities()
	{
		Capabilities result = super.getCapabilities();
		result.disableAll();

		// attributes
		result.enable(Capability.NOMINAL_ATTRIBUTES);
		result.enable(Capability.NUMERIC_ATTRIBUTES);
		result.enable(Capability.DATE_ATTRIBUTES);
		result.enable(Capability.MISSING_VALUES);

		// class
		result.enable(Capability.NOMINAL_CLASS);
		result.enable(Capability.MISSING_CLASS_VALUES);

		// instances
		result.setMinimumNumberInstances(0);

		return result;
	}

	/**
	 * Generates the classifier.
	 * 
	 * @param instances
	 *            set of instances serving as training data
	 * @throws Exception
	 *             if the classifier has not been generated successfully
	 */
	public void buildClassifier(Instances instances) throws Exception
	{
		//IB1 不能对字符串属性进行学习，因为这种属性不好定义距离
		// can classifier handle the data?
		getCapabilities().testWithFail(instances);

		//然类别缺失的样本抛弃。
		// remove instances with missing class
		instances = new Instances(instances);
		instances.deleteWithMissingClass();
		
		//保存训练样本
		m_Train = new Instances(instances, 0, instances.numInstances());

		//m_MinArray 和 m_MaxArray 分别保存每一个属性的最小值和最大值。
		m_MinArray = new double[m_Train.numAttributes()];
		m_MaxArray = new double[m_Train.numAttributes()];
		for (int i = 0; i < m_Train.numAttributes(); i++)
		{
			//NaN(不是任何数字) 不等于任何浮点数值，包括它自身在内
			m_MinArray[i] = m_MaxArray[i] = Double.NaN;
		}
		Enumeration enu = m_Train.enumerateInstances();
		while (enu.hasMoreElements())
		{
			updateMinMax((Instance) enu.nextElement());
		}
	}

	/**
	 * Updates the classifier.
	 * 
	 * @param instance
	 *            the instance to be put into the classifier
	 * @throws Exception
	 *             if the instance could not be included successfully
	 */
	public void updateClassifier(Instance instance) throws Exception
	{

		if (m_Train.equalHeaders(instance.dataset()) == false)
		{
			throw new Exception("Incompatible instance types");
		}
		if (instance.classIsMissing())
		{
			return;
		}
		m_Train.add(instance);
		updateMinMax(instance);
	}

	/**
	 * Classifies the given test instance.
	 * 
	 * @param instance	  the instance to be classified
	 * @return the predicted class for the instance
	 */
	public double classifyInstance(Instance instance) throws Exception
	{
		if (m_Train.numInstances() == 0)
		{
			throw new Exception("No training instances!");
		}

		double distance, minDistance = Double.MAX_VALUE, classValue = 0;
		//因为要进化归范化，所以对这个分类的样本再次调用 updateMinMax。
		updateMinMax(instance);
		//然后对训练样本进行循环，用 distance 计算与每一个样本的距离，如果比前面的距离小，则记录，最后返回
		//与测试样本距离最小的样本的类别值。
		Enumeration enu = m_Train.enumerateInstances();
		while (enu.hasMoreElements())
		{
			Instance trainInstance = (Instance) enu.nextElement();
			if (!trainInstance.classIsMissing())
			{
				distance = distance(instance, trainInstance);
				if (distance < minDistance)
				{
					minDistance = distance;
					classValue = trainInstance.classValue();
				}
			}
		}
		return classValue;
	}

	/**
	 * Calculates the distance between two instances
	 * 
	 * @param first
	 *            the first instance
	 * @param second
	 *            the second instance
	 * @return the distance between the two given instances
	 */
	private double distance(Instance first, Instance second)
	{
		double diff, distance = 0;

		for (int i = 0; i < m_Train.numAttributes(); i++)
		{
			if (i == m_Train.classIndex())
			{
				continue;
			}
			if (m_Train.attribute(i).isNominal())
			{

				// If attribute is nominal
				if (first.isMissing(i) || second.isMissing(i)
						|| ((int) first.value(i) != (int) second.value(i)))
				{
					//对离散属性来说，两个样本任一在对应属性上为缺失值，
					//距离为 1，不相等相然还是为 1。
					distance += 1;
				}
			} 
			else
			{
				// If attribute is numeric
				if (first.isMissing(i) || second.isMissing(i))
				{
					//如果两个都是缺失值，距离为 1
					if (first.isMissing(i) && second.isMissing(i))
					{
						diff = 1;
					}
					else
					{
						//其中之一在对应属性上为缺失值，把另一个不为缺失值的属性值规范化，
						//距离为 1-diff
						if (second.isMissing(i))
						{
							diff = norm(first.value(i), i);
						} 
						else
						{
							diff = norm(second.value(i), i);
						}
						if (diff < 0.5)
						{
							//设到可能的最远(当然那个缺失值比 m_MinArray，m_MaxArray 还小还大，这就不对了)
							diff = 1.0 - diff;
						}
					}
				} 
				else
				{
					diff = norm(first.value(i), i) - norm(second.value(i), i);
				}
				//如果两个都有值，就把距离相加，最后平方。
				distance += diff * diff;
			}
		}
		return distance;
	}

	/**
	 * Normalizes a given value of a numeric attribute.
	 * 
	 * @param x
	 *            the value to be normalized
	 * @param i
	 *            the attribute's index
	 * @return the normalized value
	 */
	private double norm(double x, int i)
	{

		if (Double.isNaN(m_MinArray[i])
				|| Utils.eq(m_MaxArray[i], m_MinArray[i]))
		{
			return 0;
		} 
		else
		{
			return (x - m_MinArray[i]) / (m_MaxArray[i] - m_MinArray[i]);
		}
	}

	/**
	 * Updates the minimum and maximum values for all the attributes based on a
	 * new instance.
	 * 
	 * @param instance	the new instance
	 *            
	 */
	private void updateMinMax(Instance instance)
	{
		for (int j = 0; j < m_Train.numAttributes(); j++)
		{
			if ((m_Train.attribute(j).isNumeric()) && (!instance.isMissing(j)))
			{
				//Double.isNaN(m_MinArray[j])判断是不是 m_MinArray 和 m_MaxArray 
				//已经赋值过了
				//如果数组中的值不是一个数字
				if (Double.isNaN(m_MinArray[j]))
				{
					m_MinArray[j] = instance.value(j);
					m_MaxArray[j] = instance.value(j);
				}
				else
				{
					if (instance.value(j) < m_MinArray[j])
					{
						m_MinArray[j] = instance.value(j);
					} 
					else
					{
						if (instance.value(j) > m_MaxArray[j])
						{
							m_MaxArray[j] = instance.value(j);
						}
					}
				}
			}
		}
	}

	public static void main(String[] argv)
	{
		runClassifier(new K1NN(), argv);
	}
}
