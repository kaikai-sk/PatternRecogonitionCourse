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

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.Enumeration;
import java.util.HashMap;
import java.util.Hashtable;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

import javax.xml.crypto.Data;

import java_cup.internal_error;

public class Id3_splitAOverAvg extends Classifier implements TechnicalInformationHandler,
Sourcable
{
	/** The node's successors. 	当前结点的子结点*/
	private Id3_splitAOverAvg[] m_Successors;

	/** Attribute used for splitting. 
	 * m_Attribute 是分裂属性*/
	private Attribute m_Attribute;

	/** Class value if node is leaf. */
	private double m_ClassValue;

	/** Class distribution if node is leaf. 
	 *Distribution 表示的是这个结点属于某个类别的概率，如 m_Distribution[0] == 0.1 表示当
		前结点属于类别 0 的概率为 0.1
	 */
	private double[] m_Distribution;

	/** Class attribute of dataset. */
	private Attribute m_ClassAttribute;
	//训练样本
	private Instances m_Instances;
	private ArrayList<Integer> indexOfUsedAtts=new ArrayList<Integer>();
	
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
		 * getCapabilities().testWithFail(data)是判断是否 ID3 能处理选择的数据集，比如什么连续
		 *属性，类别索引没有设置等等。
		 */
		getCapabilities().testWithFail(data);

		// remove instances with missing class
		m_Instances = new Instances(data);
		/**
		 * 而 data.deleteWithMissingClass 则是删除有缺失样本的函数
		 */
		m_Instances.deleteWithMissingClass();

		makeTree(m_Instances);
	}

	/**
	 * Method for building an Id3 tree.
	 * 
	 * @param data
	 *            the training data
	 * @exception Exception
	 *                if decision tree can't be built successfully
	 */
	private void makeTree(Instances data) throws Exception
	{
		m_Instances=new Instances(data);
		// Check if no instances have reached this node.
		// 如果数据集中的样本数量是零
		if (m_Instances.numInstances() == 0)
		{
			m_Attribute = null;
			m_ClassValue = Instance.missingValue();
			m_Distribution = new double[data.numClasses()];
			return;
		}

		// Compute attribute with maximum information gain.
		double[] infoGains = new double[data.numAttributes()];
		double[] splitAs=new double[data.numAttributes()];
		for(int i=0;i<splitAs.length;i++)
		{
			splitAs[i]=-1;
		}
		//Returns an enumeration of all the attributes. The class attribute (if set) is skipped by this enumeration.
		//返回所有属性的枚举器。如果设置了分类属性，会跳过这个是属性
		Enumeration attEnum = data.enumerateAttributes();
		while (attEnum.hasMoreElements())
		{
			Attribute att = (Attribute) attEnum.nextElement();
			infoGains[att.index()] = computeInfoGain(data, att);
			splitAs[att.index()]=computeSplitA(data, att);
		}
		/**
		 * Util.maxIndex 返回信息增益最大的下标，这个属性作为分裂属性保存在
		 * m_Attribute 成员变量中。
		 */
		int maxIndex=maxIndexOfGainRation(infoGains,splitAs);
		m_Attribute = data.attribute(maxIndex);
		indexOfUsedAtts.add(maxIndex);
		// Make leaf if information gain is zero.
		// Otherwise create successors.
		/**
		 * 当信息增益等于 0 时为叶子结点
		 */
		if (Utils.eq(infoGains[m_Attribute.index()], 0))
		{
			//m_Attribute = null，已经不用再分裂了，所以为 null
			m_Attribute = null;
			m_Distribution = new double[data.numClasses()];//Returns the number of class labels.
			Enumeration instEnum = data.enumerateInstances();
			while (instEnum.hasMoreElements())
			{
				Instance inst = (Instance) instEnum.nextElement();
				/**
				 * 也就是将每个样本的相应的下标加 1，比如当前
				 *	叶 子 结 点 有 10 个 样 本 ， 9 个 属 于 第 一 个 类 别 ， 1 个 属 于 第 五 个 类 别 ， 则
				 *	m_Distribution[0]=9,m_Distribution[ 4]=1。
				 */
				m_Distribution[(int) inst.classValue()]++;
			}
			//简 单 地 理 解 为 归 一 化 
			Utils.normalize(m_Distribution);
			//属于哪个类别的概率最高，那当然就是哪个类别，
			m_ClassValue = Utils.maxIndex(m_Distribution);
			//类别属性。
			m_ClassAttribute = data.classAttribute();
		}
		else
		{
			/**
			 * 在 splitData(data, m_Attribute)函数中，data 被
        	 * 分裂成 m_Attribute 离散值个子结点，比如 m_Attribute 有三种取值”green”，”red”，”blue”，
		     * 则 splitData 将 data 分成 3 部分到 splitData 中。
			 */
			Instances[] splitData = splitData(data, m_Attribute);
			m_Successors = new Id3_splitAOverAvg[m_Attribute.numValues()];
			for (int j = 0; j < m_Attribute.numValues(); j++)
			{
				m_Successors[j] = new Id3_splitAOverAvg();
				m_Successors[j].makeTree(splitData[j]);
			}
		}
	}

	/**
	 * 得到具有组大信息增益比例的属性的下标
	 * @param infoGains  所有子节点的infgains数组
	 * @param splitAs	  所有	的属性情况的split数组
	 * @return 具有最大信息增益比的属性的下标
	 */
	private int maxIndexOfGainRation(double infoGains[],double splitAs[])
	{
		//返回所有属性的枚举器。如果设置了分类属性，会跳过这个是属性
		Enumeration attEnum = m_Instances.enumerateAttributes();
		double maxGainRatio=0;//最大信息增益率
		int maxIndex=0;//最大信息增益率的下标
		
		int attNum=0;//属性的个数
		double sumIGain=0.0;//总的信息增益率
		//计算总得信息增益
		while(attEnum.hasMoreElements())
		{
			Attribute attribute=(Attribute)attEnum.nextElement();
			int attIndex=attribute.index();
			sumIGain+=infoGains[attIndex];
			attNum++;
		}
		
		//选择能够得到最大增益率的属性 ,而且那个属性的信
		//息增益至少要等于所有属性的信息增益的平均值 
		attEnum = m_Instances.enumerateAttributes();
		int attNum2=0;
		while (attEnum.hasMoreElements())
		{
			Attribute attribute=(Attribute)attEnum.nextElement();
			int attIndex=attribute.index();
			double gainRatio=(double)(infoGains[attIndex])/(double)(splitAs[attIndex]);
			if(gainRatio>maxGainRatio && infoGains[attIndex]>(sumIGain/attNum))
			{
				maxGainRatio = gainRatio;
				maxIndex = attIndex;
				attNum2++;
			}
		}
		if(attNum2>0)
		{
			return maxIndex;
		}
		else 
		{
			attEnum = m_Instances.enumerateAttributes();
			while(attEnum.hasMoreElements())
			{
				Attribute attribute=(Attribute)attEnum.nextElement();
				int attIndex=attribute.index();
				double gainRatio=(double)(infoGains[attIndex])/(double)(splitAs[attIndex]);
				if(gainRatio>maxGainRatio)
				{
					maxGainRatio = gainRatio;
					maxIndex = attIndex;
				}
			}
			return maxIndex;
		}
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
	 * 计算SplitA值，用于计算信息信息增益比例
	 * @param data 要计算SplitA的训练练样本的集合
	 * @param attribute 要计算SplitA的属性
	 * @return
	 */
	private double computeSplitA(Instances data,Attribute attribute)
	{
		int valCounts[]=new int[attribute.numValues()];
		//统计出每个属性值各有多少个样本
		//遍历样本集合data的迭代器
		Enumeration instEnum = data.enumerateInstances();
		while (instEnum.hasMoreElements())
		{
			Instance inst = (Instance) instEnum.nextElement();
			String value=inst.stringValue(attribute);
			valCounts[attribute.indexOfValue(value)]++;
		}
		
		double splitA = 0.0;
	  	 int num=data.numInstances();
		for (int j = 0; j < attribute.numValues(); j++)
		{
			if(valCounts[j]>0)
			{
				splitA -= valCounts[j] * Utils.log2(valCounts[j]);
			}
		}
		splitA /=(double)num;
		return splitA + Utils.log2(data.numInstances()); 			
	}
	
	/**
	 * 计算数据集的熵
	 * @param data 要计算熵的数据集
	 * @return 数据集的分类属性的熵
	 * @throws Exception 计算失败之后抛出异常
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
	 * 将 data分裂成 att.numValues()个子结点，inst.value(att)就是根据 inst样本 att 属性值将 inst
	 *    样本分成相应的子结点中。（确切点，也不是子结点，一个 Instances 数组元素）
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
		} 
		else
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
	 * Returns a string describing the classifier.
	 * 
	 * @return a description suitable for the GUI.
	 */
	public String globalInfo()
	{

		return "Class for constructing an unpruned decision tree based on the ID3 "
				+ "algorithm. Can only deal with nominal attributes. No missing values "
				+ "allowed. Empty leaves may result in unclassified instances. For more "
				+ "information see: \n\n"
				+ getTechnicalInformation().toString();
	}

	/**
	 * Returns an instance of a TechnicalInformation object, containing detailed
	 * information about the technical background of this class, e.g., paper
	 * reference or book this class is based on.
	 * 
	 * @return the technical information about this class
	 */
	public TechnicalInformation getTechnicalInformation()
	{
		TechnicalInformation result;

		result = new TechnicalInformation(Type.ARTICLE);
		result.setValue(Field.AUTHOR, "R. Quinlan");
		result.setValue(Field.YEAR, "1986");
		result.setValue(Field.TITLE, "Induction of decision trees");
		result.setValue(Field.JOURNAL, "Machine Learning");
		result.setValue(Field.VOLUME, "1");
		result.setValue(Field.NUMBER, "1");
		result.setValue(Field.PAGES, "81-106");

		return result;
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

	
	public static void main(String[] args)
	{
		runClassifier(new Id3_splitAOverAvg(), args);
	}

}
