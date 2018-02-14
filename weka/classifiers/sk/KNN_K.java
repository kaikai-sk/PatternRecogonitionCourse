package weka.classifiers.sk;

import weka.classifiers.*;
import weka.core.*;
import weka.core.neighboursearch.LinearNNSearch;
import weka.core.neighboursearch.NearestNeighbourSearch;

import java.util.*;

/**
 * Implement an KNN classifier.
 */
public class KNN_K extends Classifier
{

	/** The training instances used for classification. */
	private Instances m_Train;

	/** The number of neighbours to use for classification. */
	private int m_kNN;

	// k值的上界
	protected int m_kNNUpper;
	// 用来找邻居的
	protected NearestNeighbourSearch m_NNSearch = new LinearNNSearch();
	boolean m_MeanSquared = false;
	protected int m_NumClasses;
	/** The class attribute type. */
	protected int m_ClassType;
	/** no weighting. */
	public static final int WEIGHT_NONE = 1;
	/** weight by 1/distance. */
	public static final int WEIGHT_INVERSE = 2;
	/** weight by 1-distance. */
	public static final int WEIGHT_SIMILARITY = 4;
	/** Whether the neighbours should be distance-weighted. */
	protected int m_DistanceWeighting = 2;
	protected double m_NumAttributesUsed;

	/**
	 * Builds KNN classifier.
	 * 
	 * @param data
	 *            the training data
	 * @exception Exception
	 *                if classifier can't be built successfully
	 */
	public void buildClassifier(Instances data) throws Exception
	{
		m_Train = new Instances(data);
		m_NumClasses = data.numClasses();
		m_ClassType = data.classAttribute().type();
		m_kNN = 10;
		m_kNNUpper = 11;// (int) Math.sqrt(data.numInstances())+1;
		m_NumAttributesUsed = 0.0;
		for (int i = 0; i < m_Train.numAttributes(); i++)
		{
			if ((i != m_Train.classIndex())
					&& (m_Train.attribute(i).isNominal() || m_Train
							.attribute(i).isNumeric()))
			{
				m_NumAttributesUsed += 1.0;
			}
		}
	}

	protected void crossValidate()
	{

		try
		{
			double[] performanceStats = new double[m_kNNUpper];
			double[] performanceStatsSq = new double[m_kNNUpper];

			for (int i = 0; i < m_kNNUpper; i++)
			{
				performanceStats[i] = 0;
				performanceStatsSq[i] = 0;
			}

			m_kNN = m_kNNUpper;
			Instance instance;
			Instances neighbours;
			double[] origDistances, convertedDistances;
			for (int i = 0; i < m_Train.numInstances(); i++)
			{
				instance = m_Train.instance(i);
				NeighborList neighborList=findNeighbors(instance, m_kNN);
				neighbours=neighborList.getNeiborInstances();
				origDistances = getDistances(instance,neighbours);

				for (int j = m_kNNUpper - 1; j >= 0; j--)
				{
					convertedDistances = new double[origDistances.length];
					System.arraycopy(origDistances, 0, convertedDistances, 0,
							origDistances.length);
					double[] distribution = makeDistribution(neighbours,
							convertedDistances);
					double thisPrediction = Utils.maxIndex(distribution);
					if (m_Train.classAttribute().isNumeric())
					{
						thisPrediction = distribution[0];
						double err = thisPrediction - instance.classValue();
						performanceStatsSq[j] += err * err; // Squared error
						performanceStats[j] += Math.abs(err); // Absolute error
					} else
					{
						if (thisPrediction != instance.classValue())
						{
							performanceStats[j]++; // Classification error
						}
					}
					if (j >= 1)
					{
						neighbours = pruneToK(neighbours, convertedDistances, j);
					}
				}
			}
			
			// Check through the performance stats and select the best
			// k value (or the lowest k if more than one best)
			double[] searchStats = performanceStats;
			if (m_Train.classAttribute().isNumeric() && m_MeanSquared)
			{
				searchStats = performanceStatsSq;
			}
			double bestPerformance = Double.NaN;
			int bestK = 1;
			for (int i = 0; i < m_kNNUpper; i++)
			{
				if (Double.isNaN(bestPerformance)
						|| (bestPerformance > searchStats[i]))
				{
					bestPerformance = searchStats[i];
					bestK = i + 1;
				}
			}
			m_kNN = bestK;
			if (m_Debug)
			{
				System.err.println("Selected k = " + bestK);
			}

		} catch (Exception ex)
		{
			throw new Error("Couldn't optimize by cross-validation: "
					+ ex.getMessage());
		}
	}

	/**
	 * 拿到样本到所有邻居的距离数组
	 * 
	 * @param instance
	 * @param neiborInstances
	 * @return
	 */
	public double[] getDistances(Instance instance, Instances neiborInstances)
	{
		double[] distances = new double[neiborInstances.numInstances()];
		for (int i = 0; i < neiborInstances.numInstances(); i++)
		{
			distances[i] = distance(instance, neiborInstances.instance(i));
		}
		return distances;
	}

	/**
	 * Prunes the list to contain the k nearest neighbors. If there are multiple
	 * neighbors at the k'th distance, all will be kept.
	 * 
	 * @param neighbours
	 *            the neighbour instances.
	 * @param distances
	 *            the distances of the neighbours from target instance.
	 * @param k
	 *            the number of neighbors to keep.
	 * @return the pruned neighbours.
	 */
	public Instances pruneToK(Instances neighbours, double[] distances, int k)
	{

		if (neighbours == null || distances == null
				|| neighbours.numInstances() == 0)
		{
			return null;
		}
		if (k < 1)
		{
			k = 1;
		}

		int currentK = 0;
		double currentDist;
		for (int i = 0; i < neighbours.numInstances(); i++)
		{
			currentK++;
			currentDist = distances[i];
			if (currentK > k && currentDist != distances[i - 1])
			{
				currentK--;
				neighbours = new Instances(neighbours, 0, currentK);
				break;
			}
		}

		return neighbours;
	}

	/**
	 * Turn the list of nearest neighbors into a probability distribution.
	 * 
	 * @param neighbours
	 *            the list of nearest neighboring instances
	 * @param distances
	 *            the distances of the neighbors
	 * @return the probability distribution
	 * @throws Exception
	 *             if computation goes wrong or has no class attribute
	 */
	protected double[] makeDistribution(Instances neighbours, double[] distances)
			throws Exception
	{

		double total = 0, weight;
		double[] distribution = new double[m_NumClasses];

		// Set up a correction to the estimator
		if (m_ClassType == Attribute.NOMINAL)
		{
			for (int i = 0; i < m_NumClasses; i++)
			{
				distribution[i] = 1.0 / Math.max(1, m_Train.numInstances());
			}
			total = (double) m_NumClasses / Math.max(1, m_Train.numInstances());
		}

		for (int i = 0; i < neighbours.numInstances(); i++)
		{
			// Collect class counts
			Instance current = neighbours.instance(i);
			distances[i] = distances[i] * distances[i];
			distances[i] = Math.sqrt(distances[i] / m_NumAttributesUsed);
			switch (m_DistanceWeighting)
			{
				case WEIGHT_INVERSE:
					weight = 1.0 / (distances[i] + 0.001); // to avoid div by
															// zero
					break;
				case WEIGHT_SIMILARITY:
					weight = 1.0 - distances[i];
					break;
				default: // WEIGHT_NONE:
					weight = 1.0;
					break;
			}
			weight *= current.weight();
			try
			{
				switch (m_ClassType)
				{
					case Attribute.NOMINAL:
						distribution[(int) current.classValue()] += weight;
						break;
					case Attribute.NUMERIC:
						distribution[0] += current.classValue() * weight;
						break;
				}
			} catch (Exception ex)
			{
				throw new Error("Data has no class attribute!");
			}
			total += weight;
		}

		// Normalise distribution
		if (total > 0)
		{
			Utils.normalize(distribution, total);
		}
		return distribution;
	}

	/**
	 * Computes class distribution for a test instance.
	 * 
	 * @param instance
	 *            the instance for which distribution is to be computed
	 * @return the class distribution for the given instance
	 */
	public double[] distributionForInstance(Instance instance) throws Exception
	{
		// crossValidate简单地说就是用蛮力找在到底用多少个邻居好，
		// 它对m_Train中的样本进行循环，对每个样本找邻居，然后统计看寻找多少个邻居时最好。
		crossValidate();
		System.err.println("m_kNN:" + m_kNN);
		NeighborList neighborlist = findNeighbors(instance, m_kNN);
		return computeDistribution(neighborInstances(neighborlist), instance);
	}

	/**
	 * Build the list of nearest k neighbors to the given test instance.
	 * 
	 * @param instance
	 *            the instance to search for neighbours
	 * @return a list of neighbors
	 */
	private NeighborList findNeighbors(Instance instance, int kNN)
	{

		double distance;
		NeighborList neighborlist = new NeighborList(kNN);
		for (int i = 0; i < m_Train.numInstances(); i++)
		{
			Instance trainInstance = m_Train.instance(i);
			distance = distance(instance, trainInstance);
			if (neighborlist.isEmpty() || i < kNN
					|| distance <= neighborlist.m_Last.m_Distance)
			{
				neighborlist.insertSorted(distance, trainInstance);
			}
		}
		return neighborlist;

	}

	/**
	 * Turn the list of nearest neighbors into a probability distribution
	 * 
	 * @param neighborlist
	 *            the list of nearest neighboring instances
	 * @return the probability distribution
	 */
	private Instances neighborInstances(NeighborList neighborlist)
			throws Exception
	{

		Instances neighborInsts = new Instances(m_Train,
				neighborlist.currentLength());
		if (!neighborlist.isEmpty())
		{
			NeighborNode current = neighborlist.m_First;
			while (current != null)
			{
				neighborInsts.add(current.m_Instance);
				current = current.m_Next;
			}
		}
		return neighborInsts;

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

		double distance = 0;
		for (int i = 0; i < m_Train.numAttributes(); i++)
		{
			if (i == m_Train.classIndex())
				continue;
			if ((int) first.value(i) != (int) second.value(i))
			{
				distance += 1;
			}
		}
		return distance;
	}

	/**
	 * Compute the distribution.
	 * 
	 * @param data
	 *            the training data
	 * @exception Exception
	 *                if classifier can't be built successfully
	 */
	private double[] computeDistribution(Instances data, Instance instance)
			throws Exception
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
			System.out.println(Evaluation.evaluateModel(new KNN_K(), args));
		} catch (Exception e)
		{
			System.err.println(e.getMessage());
		}
	}

	/*
	 * A class for storing data about a neighboring instance
	 */
	private class NeighborNode
	{

		/** The neighbor instance */
		private Instance m_Instance;

		public Instance getM_Instance()
		{
			return m_Instance;
		}

		public void setM_Instance(Instance m_Instance)
		{
			this.m_Instance = m_Instance;
		}

		public double getM_Distance()
		{
			return m_Distance;
		}

		public void setM_Distance(double m_Distance)
		{
			this.m_Distance = m_Distance;
		}

		public NeighborNode getM_Next()
		{
			return m_Next;
		}

		public void setM_Next(NeighborNode m_Next)
		{
			this.m_Next = m_Next;
		}

		/** The distance from the current instance to this neighbor */
		private double m_Distance;

		/** A link to the next neighbor instance */
		private NeighborNode m_Next;

		/**
		 * Create a new neighbor node.
		 * 
		 * @param distance
		 *            the distance to the neighbor
		 * @param instance
		 *            the neighbor instance
		 * @param next
		 *            the next neighbor node
		 */
		public NeighborNode(double distance, Instance instance,
				NeighborNode next)
		{
			m_Distance = distance;
			m_Instance = instance;
			m_Next = next;
		}

		/**
		 * Create a new neighbor node that doesn't link to any other nodes.
		 * 
		 * @param distance
		 *            the distance to the neighbor
		 * @param instance
		 *            the neighbor instance
		 */
		public NeighborNode(double distance, Instance instance)
		{

			this(distance, instance, null);
		}
	}

	/*
	 * A class for a linked list to store the nearest k neighbours to an
	 * instance.
	 */
	private class NeighborList
	{

		/** The first node in the list */
		private NeighborNode m_First;

		/** The last node in the list */
		private NeighborNode m_Last;

		/** The number of nodes to attempt to maintain in the list */
		private int m_Length = 1;

		public Instances getNeiborInstances()
		{
			Instances neiborInstances = new Instances(m_Train);
			neiborInstances.delete();
			NeighborNode current = m_First;
			while (current != null)
			{
				neiborInstances.add(current.getM_Instance());
				current = current.m_Next;
			}
			return neiborInstances;
		}

		/**
		 * Creates the neighborlist with a desired length
		 * 
		 * @param length
		 *            the length of list to attempt to maintain
		 */
		public NeighborList(int length)
		{

			m_Length = length;
		}

		/**
		 * Gets whether the list is empty.
		 * 
		 * @return true if so
		 */
		public boolean isEmpty()
		{

			return (m_First == null);
		}

		/**
		 * Gets the current length of the list.
		 * 
		 * @return the current length of the list
		 */
		public int currentLength()
		{

			int i = 0;
			NeighborNode current = m_First;
			while (current != null)
			{
				i++;
				current = current.m_Next;
			}
			return i;
		}

		/**
		 * Inserts an instance neighbor into the list, maintaining the list
		 * sorted by distance.
		 * 
		 * @param distance
		 *            the distance to the instance
		 * @param instance
		 *            the neighboring instance
		 */
		public void insertSorted(double distance, Instance instance)
		{

			if (isEmpty())
			{
				m_First = m_Last = new NeighborNode(distance, instance);
			} else
			{
				NeighborNode current = m_First;
				if (distance < m_First.m_Distance)
				{// Insert at head
					m_First = new NeighborNode(distance, instance, m_First);
				} else
				{ // Insert further down the list
					for (; (current.m_Next != null)
							&& (current.m_Next.m_Distance < distance); current = current.m_Next)
						;
					current.m_Next = new NeighborNode(distance, instance,
							current.m_Next);
					if (current.equals(m_Last))
					{
						m_Last = current.m_Next;
					}
				}

				// Trip down the list until we've got k list elements (or more
				// if the distance to the last elements is the same).
				int valcount = 0;
				for (current = m_First; current.m_Next != null; current = current.m_Next)
				{
					valcount++;
					if ((valcount >= m_Length)
							&& (current.m_Distance != current.m_Next.m_Distance))
					{
						m_Last = current;
						current.m_Next = null;
						break;
					}
				}
			}
		}

		/**
		 * Prunes the list to contain the k nearest neighbors. If there are
		 * multiple neighbors at the k'th distance, all will be kept.
		 * 
		 * @param k
		 *            the number of neighbors to keep in the list.
		 */
		public void pruneToK(int k)
		{

			if (isEmpty())
			{
				return;
			}
			if (k < 1)
			{
				k = 1;
			}
			int currentK = 0;
			double currentDist = m_First.m_Distance;
			NeighborNode current = m_First;
			for (; current.m_Next != null; current = current.m_Next)
			{
				currentK++;
				currentDist = current.m_Distance;
				if ((currentK >= k)
						&& (currentDist != current.m_Next.m_Distance))
				{
					m_Last = current;
					current.m_Next = null;
					break;
				}
			}
		}

	}

}
