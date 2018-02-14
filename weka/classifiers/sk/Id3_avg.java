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
 *    Id3.java
 *    Copyright (C) 1999 University of Waikato, Hamilton, New Zealand
 *
 */

package weka.classifiers.sk;

import weka.classifiers.Classifier;
import weka.classifiers.Sourcable;
import weka.core.Attribute;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.NoSupportForMissingValuesException;
import weka.core.RevisionUtils;
import weka.core.TechnicalInformation;
import weka.core.TechnicalInformationHandler;
import weka.core.Utils;
import weka.core.Capabilities.Capability;
import weka.core.TechnicalInformation.Field;
import weka.core.TechnicalInformation.Type;

import java.util.Enumeration;

import java_cup.internal_error;

/**
 * ��ID3�㷨�Ļ����Ͻ��иĽ�����ѡ��IGain���ģ�ѡ��Igain/��������
 * @author ����
 *
 */
public class Id3_avg extends Classifier
{
	/** The node's successors. 	��ǰ�����ӽ��*/
	private Id3_avg[] m_Successors;

	/** Attribute used for splitting. 
	 * m_Attribute �Ƿ�������*/
	private Attribute m_Attribute;

	/** Class value if node is leaf. */
	private double m_ClassValue;

	/** Class distribution if node is leaf. 
	 *Distribution ��ʾ��������������ĳ�����ĸ��ʣ��� m_Distribution[0] == 0.1 ��ʾ��
		ǰ���������� 0 �ĸ���Ϊ 0.1
	 */
	private double[] m_Distribution;

	/** Class attribute of dataset. */
	private Attribute m_ClassAttribute;
	
	//��¼�������Ե�ȡֵ�ĸ��������磺outlook{sunny��rain,overcaset} 3
	int[] attNum0fVal;

	/**
	 * Returns default capabilities of the classifier.
	 * 
	 * @return the capabilities of this classifier
	 */
	public Capabilities getCapabilities()
	{
		Capabilities result = super.getCapabilities();
		result.disableAll();

		// attributes
		result.enable(Capability.NOMINAL_ATTRIBUTES);

		// class
		result.enable(Capability.NOMINAL_CLASS);
		result.enable(Capability.MISSING_CLASS_VALUES);

		// instances
		result.setMinimumNumberInstances(0);

		return result;
	}

	/**
	 * Builds Id3 decision tree classifier.
	 * 
	 * @param data
	 *            the training data
	 * @exception Exception
	 *                if classifier can't be built successfully
	 */
	public void buildClassifier(Instances data) throws Exception
	{

		// can classifier handle the data?
		/**
		 * getCapabilities().testWithFail(data)���ж��Ƿ� ID3 �ܴ���ѡ������ݼ�������ʲô����
		 *���ԣ��������û�����õȵȡ�
		 */
		getCapabilities().testWithFail(data);

		// remove instances with missing class
		data = new Instances(data);
		/**
		 * �� data.deleteWithMissingClass ����ɾ����ȱʧ�����ĺ���
		 */
		data.deleteWithMissingClass();

		makeTree(data);
	}

	/**
	 * ����ID3������
	 * @param data ѵ������
	 * @exception û�гɹ��������ʱ���׳��쳣
	 */
	private void makeTree(Instances data) throws Exception
	{
		// Check if no instances have reached this node.
		// ������ݼ��е�������������
		if (data.numInstances() == 0)
		{
			m_Attribute = null;
			m_ClassValue = Instance.missingValue();
			m_Distribution = new double[data.numClasses()];
			return;
		}

		// ������������Ϣ��������ԡ�infoGains[]�洢�������Ե�IGain
		double[] infoGains = new double[data.numAttributes()];
		attNum0fVal=new int[data.numAttributes()];
		//�����������Ե�ö��������������˷������ԣ����������������
		Enumeration attEnum = data.enumerateAttributes();
		while (attEnum.hasMoreElements())
		{
			Attribute att = (Attribute) attEnum.nextElement();
			infoGains[att.index()] = computeInfoGain(data, att);
			attNum0fVal[att.index()]=att.numValues();
		}
		/**
		 * Util.maxIndex ������Ϣ���������±꣬���������Ϊ�������Ա�����
		 * m_Attribute ��Ա�����С�
		 */
		m_Attribute = data.attribute(maxIndexAfterDivAttvalnum(infoGains));

		/**
		 * Make leaf if information gain is zero.
		 *  Otherwise create successors.
		 * ����Ϣ������� 0 ʱΪҶ�ӽ��
		 */
		if (Utils.eq(infoGains[m_Attribute.index()], 0))
		{
			//m_Attribute = null���Ѿ������ٷ����ˣ�����Ϊ null
			m_Attribute = null;
			m_Distribution = new double[data.numClasses()];//Returns the number of class labels.
			Enumeration instEnum = data.enumerateInstances();
			while (instEnum.hasMoreElements())
			{
				Instance inst = (Instance) instEnum.nextElement();
				/**
				 * Ҳ���ǽ�ÿ����������Ӧ���±�� 1�����統ǰ
				 *	Ҷ �� �� �� �� 10 �� �� �� �� 9 �� �� �� �� һ �� �� �� �� 1 �� �� �� �� �� �� �� �� �� ��
				 *	m_Distribution[0]=9,m_Distribution[ 4]=1��
				 */
				m_Distribution[(int) inst.classValue()]++;
			}
			//�� �� �� �� �� Ϊ �� һ �� �� 
			Utils.normalize(m_Distribution);
			//�����ĸ����ĸ�����ߣ��ǵ�Ȼ�����ĸ����
			m_ClassValue = Utils.maxIndex(m_Distribution);
			//������ԡ�
			m_ClassAttribute = data.classAttribute();
		}
		else
		{
			/**
			 * �� splitData(data, m_Attribute)�����У�data ��
        		*���ѳ� m_Attribute ��ɢֵ���ӽ�㣬���� m_Attribute ������ȡֵ��green������red������blue����
		    	*�� splitData �� data �ֳ� 3 ���ֵ� splitData �С�
			 */
			Instances[] splitData = splitData(data, m_Attribute);
			m_Successors = new Id3_avg[m_Attribute.numValues()];
			for (int j = 0; j < m_Attribute.numValues(); j++)
			{
				m_Successors[j] = new Id3_avg();
				m_Successors[j].makeTree(splitData[j]);
			}
		}
		//</editor-fold>
	}

	/**
	 * ��Igain/�����Զ�Ӧ��ȡֵ�ĸ���֮��ȥ���ֵ���±�
	 * @return ���ŵ����Ե��±�
	 */
	private int maxIndexAfterDivAttvalnum(double infoGains[])
	{
		double maxRes=0;
		int maxIndex=0;
		for (int i = 0; i < infoGains.length; i++)
		{
			double res=infoGains[i]/attNum0fVal[i];
			if(res>maxRes)
			{
				maxRes=res;
				maxIndex=i;
			}
		}
		return maxIndex;
	}
	
	/**
	 * Classifies a given test instance using the decision tree.
	 * 
	 * @param instance
	 *            the instance to be classified
	 * @return the classification
	 * @throws NoSupportForMissingValuesException
	 *             if instance has missing values
	 */
	public double classifyInstance(Instance instance)
			throws NoSupportForMissingValuesException
	{

		if (instance.hasMissingValue())
		{
			throw new NoSupportForMissingValuesException(
					"Id3: no missing values, " + "please.");
		}
		if (m_Attribute == null)
		{
			return m_ClassValue;
		} else
		{
			return m_Successors[(int) instance.value(m_Attribute)]
					.classifyInstance(instance);
		}
	}

	/**
	 * Computes class distribution for instance using decision tree.
	 * 
	 * @param instance
	 *            the instance for which distribution is to be computed
	 * @return the class distribution for the given instance
	 * @throws NoSupportForMissingValuesException
	 *             if instance has missing values
	 */
	public double[] distributionForInstance(Instance instance)
			throws NoSupportForMissingValuesException
	{

		if (instance.hasMissingValue())
		{
			throw new NoSupportForMissingValuesException(
					"Id3: no missing values, " + "please.");
		}
		if (m_Attribute == null)
		{
			return m_Distribution;
		} 
		else
		{
			return m_Successors[(int) instance.value(m_Attribute)]
					.distributionForInstance(instance);
		}
	}

	/**
	 * Prints the decision tree using the private toString method from below.
	 * 
	 * @return a textual description of the classifier
	 */
	public String toString()
	{

		if ((m_Distribution == null) && (m_Successors == null))
		{
			return "Id3: No model built yet.";
		}
		return "Id3\n\n" + toString(0);
	}

	/**
	 * Computes information gain for an attribute.
	 * 
	 * @param data
	 *            the data for which info gain is to be computed
	 * @param att
	 *            the attribute
	 * @return the information gain for the given attribute and data
	 * @throws Exception
	 *             if computation fails
	 */
	private double computeInfoGain(Instances data, Attribute att)
			throws Exception
	{

		double infoGain = computeEntropy(data);
		Instances[] splitData = splitData(data, att);
		for (int j = 0; j < att.numValues(); j++)
		{
			if (splitData[j].numInstances() > 0)
			{
				infoGain -= ((double) splitData[j].numInstances() / (double) data
						.numInstances()) * computeEntropy(splitData[j]);
			}
		}
		return infoGain;
	}

	/**
	 * Computes the entropy of a dataset.
	 * 
	 * @param data
	 *            the data for which entropy is to be computed
	 * @return the entropy of the data's class distribution
	 * @throws Exception
	 *             if computation fails
	 */
	private double computeEntropy(Instances data) throws Exception
	{

		double[] classCounts = new double[data.numClasses()];
		Enumeration instEnum = data.enumerateInstances();
		while (instEnum.hasMoreElements())
		{
			Instance inst = (Instance) instEnum.nextElement();
			classCounts[(int) inst.classValue()]++;
		}
		double entropy = 0;
		for (int j = 0; j < data.numClasses(); j++)
		{
			if (classCounts[j] > 0)
			{
				entropy -= classCounts[j] * Utils.log2(classCounts[j]);
			}
		}
		entropy /= (double) data.numInstances();
		return entropy + Utils.log2(data.numInstances());
	}

	/**
	 * Splits a dataset according to the values of a nominal attribute.
	 * �� data���ѳ� att.numValues()���ӽ�㣬inst.value(att)���Ǹ��� inst���� att ����ֵ�� inst
	 *    �����ֳ���Ӧ���ӽ���С���ȷ�е㣬Ҳ�����ӽ�㣬һ�� Instances ����Ԫ�أ�
	 * @param data
	 *            the data which is to be split
	 * @param att
	 *            the attribute to be used for splitting
	 * @return the sets of instances produced by the split
	 */
	private Instances[] splitData(Instances data, Attribute att)
	{

		Instances[] splitData = new Instances[att.numValues()];
		for (int j = 0; j < att.numValues(); j++)
		{
			splitData[j] = new Instances(data, data.numInstances());
		}
		Enumeration instEnum = data.enumerateInstances();
		while (instEnum.hasMoreElements())
		{
			Instance inst = (Instance) instEnum.nextElement();
			splitData[(int) inst.value(att)].add(inst);
		}
		for (int i = 0; i < splitData.length; i++)
		{
			splitData[i].compactify();
		}
		return splitData;
	}

	/**
	 * Outputs a tree at a certain level.
	 * 
	 * @param level
	 *            the level at which the tree is to be printed
	 * @return the tree as string at the given level
	 */
	private String toString(int level)
	{

		StringBuffer text = new StringBuffer();

		if (m_Attribute == null)
		{
			if (Instance.isMissingValue(m_ClassValue))
			{
				text.append(": null");
			} else
			{
				text.append(": " + m_ClassAttribute.value((int) m_ClassValue));
			}
		} else
		{
			for (int j = 0; j < m_Attribute.numValues(); j++)
			{
				text.append("\n");
				for (int i = 0; i < level; i++)
				{
					text.append("|  ");
				}
				text.append(m_Attribute.name() + " = " + m_Attribute.value(j));
				text.append(m_Successors[j].toString(level + 1));
			}
		}
		return text.toString();
	}

	/**
	 * Adds this tree recursively to the buffer.
	 * 
	 * @param id
	 *            the unqiue id for the method
	 * @param buffer
	 *            the buffer to add the source code to
	 * @return the last ID being used
	 * @throws Exception
	 *             if something goes wrong
	 */
	protected int toSource(int id, StringBuffer buffer) throws Exception
	{
		int result;
		int i;
		int newID;
		StringBuffer[] subBuffers;

		buffer.append("\n");
		buffer.append("  protected static double node" + id
				+ "(Object[] i) {\n");

		// leaf?
		if (m_Attribute == null)
		{
			result = id;
			if (Double.isNaN(m_ClassValue))
			{
				buffer.append("    return Double.NaN;");
			} else
			{
				buffer.append("    return " + m_ClassValue + ";");
			}
			if (m_ClassAttribute != null)
			{
				buffer.append(" // "
						+ m_ClassAttribute.value((int) m_ClassValue));
			}
			buffer.append("\n");
			buffer.append("  }\n");
		} else
		{
			buffer.append("    checkMissing(i, " + m_Attribute.index()
					+ ");\n\n");
			buffer.append("    // " + m_Attribute.name() + "\n");

			// subtree calls
			subBuffers = new StringBuffer[m_Attribute.numValues()];
			newID = id;
			for (i = 0; i < m_Attribute.numValues(); i++)
			{
				newID++;

				buffer.append("    ");
				if (i > 0)
				{
					buffer.append("else ");
				}
				buffer.append("if (((String) i[" + m_Attribute.index()
						+ "]).equals(\"" + m_Attribute.value(i) + "\"))\n");
				buffer.append("      return node" + newID + "(i);\n");

				subBuffers[i] = new StringBuffer();
				newID = m_Successors[i].toSource(newID, subBuffers[i]);
			}
			buffer.append("    else\n");
			buffer.append("      throw new IllegalArgumentException(\"Value '\" + i["
					+ m_Attribute.index() + "] + \"' is not allowed!\");\n");
			buffer.append("  }\n");

			// output subtree code
			for (i = 0; i < m_Attribute.numValues(); i++)
			{
				buffer.append(subBuffers[i].toString());
			}
			subBuffers = null;

			result = newID;
		}

		return result;
	}

	/**
	 * Returns a string that describes the classifier as source. The classifier
	 * will be contained in a class with the given name (there may be auxiliary
	 * classes), and will contain a method with the signature:
	 * 
	 * <pre>
	 * <code>
	 * public static double classify(Object[] i);
	 * </code>
	 * </pre>
	 * 
	 * where the array <code>i</code> contains elements that are either Double,
	 * String, with missing values represented as null. The generated code is
	 * public domain and comes with no warranty. <br/>
	 * Note: works only if class attribute is the last attribute in the dataset.
	 * 
	 * @param className
	 *            the name that should be given to the source class.
	 * @return the object source described by a string
	 * @throws Exception
	 *             if the source can't be computed
	 */
	public String toSource(String className) throws Exception
	{
		StringBuffer result;
		int id;

		result = new StringBuffer();

		result.append("class " + className + " {\n");
		result.append("  private static void checkMissing(Object[] i, int index) {\n");
		result.append("    if (i[index] == null)\n");
		result.append("      throw new IllegalArgumentException(\"Null values "
				+ "are not allowed!\");\n");
		result.append("  }\n\n");
		result.append("  public static double classify(Object[] i) {\n");
		id = 0;
		result.append("    return node" + id + "(i);\n");
		result.append("  }\n");
		toSource(id, result);
		result.append("}\n");

		return result.toString();
	}

	/**
	 * Returns the revision string.
	 * 
	 * @return the revision
	 */
	public String getRevision()
	{
		return RevisionUtils.extract("$Revision: 6404 $");
	}

	/**
	 * Main method.
	 * 
	 * @param args
	 *            the options for the classifier
	 */
	public static void main(String[] args)
	{
		runClassifier(new Id3_avg(), args);
	}
}
