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
		//IB1 ���ܶ��ַ������Խ���ѧϰ����Ϊ�������Բ��ö������
		// can classifier handle the data?
		getCapabilities().testWithFail(instances);

		//Ȼ���ȱʧ������������
		// remove instances with missing class
		instances = new Instances(instances);
		instances.deleteWithMissingClass();
		
		//����ѵ������
		m_Train = new Instances(instances, 0, instances.numInstances());

		//m_MinArray �� m_MaxArray �ֱ𱣴�ÿһ�����Ե���Сֵ�����ֵ��
		m_MinArray = new double[m_Train.numAttributes()];
		m_MaxArray = new double[m_Train.numAttributes()];
		for (int i = 0; i < m_Train.numAttributes(); i++)
		{
			//NaN(�����κ�����) �������κθ�����ֵ����������������
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
		//��ΪҪ�����鷶�������Զ��������������ٴε��� updateMinMax��
		updateMinMax(instance);
		//Ȼ���ѵ����������ѭ������ distance ������ÿһ�������ľ��룬�����ǰ��ľ���С�����¼����󷵻�
		//���������������С�����������ֵ��
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
					//����ɢ������˵������������һ�ڶ�Ӧ������Ϊȱʧֵ��
					//����Ϊ 1���������Ȼ����Ϊ 1��
					distance += 1;
				}
			} 
			else
			{
				// If attribute is numeric
				if (first.isMissing(i) || second.isMissing(i))
				{
					//�����������ȱʧֵ������Ϊ 1
					if (first.isMissing(i) && second.isMissing(i))
					{
						diff = 1;
					}
					else
					{
						//����֮һ�ڶ�Ӧ������Ϊȱʧֵ������һ����Ϊȱʧֵ������ֵ�淶����
						//����Ϊ 1-diff
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
							//�赽���ܵ���Զ(��Ȼ�Ǹ�ȱʧֵ�� m_MinArray��m_MaxArray ��С������Ͳ�����)
							diff = 1.0 - diff;
						}
					}
				} 
				else
				{
					diff = norm(first.value(i), i) - norm(second.value(i), i);
				}
				//�����������ֵ���ͰѾ�����ӣ����ƽ����
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
				//Double.isNaN(m_MinArray[j])�ж��ǲ��� m_MinArray �� m_MaxArray 
				//�Ѿ���ֵ����
				//��������е�ֵ����һ������
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
