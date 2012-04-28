package maltlibweka;

import java.util.Enumeration;
import java.util.Vector;

import weka.attributeSelection.ASEvaluation;
import weka.attributeSelection.AttributeEvaluator;
import weka.core.Capabilities;
import weka.core.Capabilities.Capability;
import weka.core.ContingencyTables;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Option;
import weka.core.OptionHandler;
import weka.core.RevisionUtils;
import weka.core.Utils;
import weka.filters.Filter;
import weka.filters.supervised.attribute.Discretize;
import weka.filters.unsupervised.attribute.NumericToBinary;

/**
 * <!-- globalinfo-start --> InfoGainAttributeEval :<br/>
 * <br/>
 * Evaluates the worth of an attribute by measuring the information gain with
 * respect to the class.<br/>
 * <br/>
 * InfoGain(Class,Attribute) = H(Class) - H(Class | Attribute).<br/>
 * <p/>
 * <!-- globalinfo-end -->
 * 
 * <!-- options-start --> Valid options are:
 * <p/>
 * 
 * <pre>
 * -M
 *  treat missing values as a seperate value.
 * </pre>
 * 
 * <pre>
 * -B
 *  just binarize numeric attributes instead 
 *  of properly discretizing them.
 * </pre>
 * 
 * <!-- options-end -->
 * 
 * @author Mark Hall (mhall@cs.waikato.ac.nz)
 * @version $Revision: 5511 $
 * @see Discretize
 * @see NumericToBinary
 */
public class BinaryInfoGainAttributeEval extends ASEvaluation implements
	AttributeEvaluator, OptionHandler {

    /** for serialization */
    private static final long serialVersionUID = 7015976717661257014L;

    /** Treat missing values as a seperate value */
    private boolean m_missing_merge;

    /** Just binarize numeric attributes */
    private boolean m_Binarize;

    /** The info gain for each attribute */
    private double[] m_InfoGains;

    /**
     * Returns a string describing this attribute evaluator
     * 
     * @return a description of the evaluator suitable for displaying in the
     *         explorer/experimenter gui
     */
    public String globalInfo() {
	return "BinaryInfoGainAttributeEval :\n\nEvaluates the worth of an attribute "
		+ "by measuring the information gain with respect to the class.\n\n"
		+ "InfoGain(Class,Attribute) = H(Class) - H(Class | Attribute).\n";
    }

    /**
     * Constructor
     */
    public BinaryInfoGainAttributeEval() {
	resetOptions();
    }

    /**
     * Returns an enumeration describing the available options.
     * 
     * @return an enumeration of all the available options.
     **/
    public Enumeration<Option> listOptions() {
	Vector<Option> newVector = new Vector<Option>();
	return newVector.elements();
    }

    public void setOptions(String[] options) throws Exception {
	resetOptions();
    }

    /**
     * Gets the current settings of WrapperSubsetEval.
     * 
     * @return an array of strings suitable for passing to setOptions()
     */
    public String[] getOptions() {
	String[] options = new String[2];
	int current = 0;

	if (!getMissingMerge()) {
	    options[current++] = "-M";
	}
	if (getBinarizeNumericAttributes()) {
	    options[current++] = "-B";
	}

	while (current < options.length) {
	    options[current++] = "";
	}

	return options;
    }

    /**
     * Returns the tip text for this property
     * 
     * @return tip text for this property suitable for displaying in the
     *         explorer/experimenter gui
     */
    public String binarizeNumericAttributesTipText() {
	return "Just binarize numeric attributes instead of properly discretizing them.";
    }

    /**
     * Binarize numeric attributes.
     * 
     * @param b
     *            true=binarize numeric attributes
     */
    public void setBinarizeNumericAttributes(boolean b) {
	m_Binarize = b;
    }

    /**
     * get whether numeric attributes are just being binarized.
     * 
     * @return true if missing values are being distributed.
     */
    public boolean getBinarizeNumericAttributes() {
	return m_Binarize;
    }

    /**
     * Returns the tip text for this property
     * 
     * @return tip text for this property suitable for displaying in the
     *         explorer/experimenter gui
     */
    public String missingMergeTipText() {
	return "Distribute counts for missing values. Counts are distributed "
		+ "across other values in proportion to their frequency. Otherwise, "
		+ "missing is treated as a separate value.";
    }

    /**
     * distribute the counts for missing values across observed values
     * 
     * @param b
     *            true=distribute missing values.
     */
    public void setMissingMerge(boolean b) {
	m_missing_merge = b;
    }

    /**
     * get whether missing values are being distributed or not
     * 
     * @return true if missing values are being distributed.
     */
    public boolean getMissingMerge() {
	return m_missing_merge;
    }

    /**
     * Returns the capabilities of this evaluator.
     * 
     * @return the capabilities of this evaluator
     * @see Capabilities
     */
    public Capabilities getCapabilities() {
	Capabilities result = super.getCapabilities();
	result.disableAll();
	// attributes
	result.enable(Capability.NUMERIC_ATTRIBUTES);
	// class
	result.enable(Capability.NOMINAL_CLASS);
	result.enable(Capability.MISSING_CLASS_VALUES);
	return result;
    }

    /**
     * Initializes an information gain attribute evaluator. All non-class
     * attributes must be binary.
     * 
     * @param data
     *            set of instances serving as training data
     * @throws Exception
     *             if the evaluator has not been generated successfully
     */
    public void buildEvaluator(Instances data) throws Exception {
	System.out.println("building evaluator...");
	getCapabilities().testWithFail(data);
	int classIndex = data.classIndex();
	int numInstances = data.numInstances();

	int numClasses = data.attribute(classIndex).numValues();

	// Reserve space and initialize counters
	double[][][] counts = new double[data.numAttributes()][][];
	for (int k = 0; k < data.numAttributes(); k++) {
	    if (k != classIndex) {
		int numValues = data.attribute(k).numValues();
		counts[k] = new double[numValues + 1][numClasses + 1];
	    }
	}
	// Initialize counters
	double[] temp = new double[numClasses + 1];
	for (int k = 0; k < numInstances; k++) {
	    Instance inst = data.instance(k);
	    if (inst.classIsMissing()) {
		temp[numClasses] += inst.weight();
	    } else {
		temp[(int) inst.classValue()] += inst.weight();
	    }
	}
	for (int k = 0; k < counts.length; k++) {
	    if (k != classIndex) {
		for (int i = 0; i < temp.length; i++) {
		    counts[k][0][i] = temp[i];
		}
	    }
	}

	// Get counts
	for (int k = 0; k < numInstances; k++) {
	    Instance inst = data.instance(k);
	    for (int i = 0; i < inst.numValues(); i++) {
		if (inst.index(i) != classIndex) {
		    counts[inst.index(i)][(int) inst.value(i)][(int) inst
		                                                     .classValue()] += inst.weight();
		    counts[inst.index(i)][0][(int) inst.classValue()] -= inst
			    .weight();
		}
	    }
	}
	// Compute info gains
	m_InfoGains = new double[data.numAttributes()];
	for (int i = 0; i < data.numAttributes(); i++) {
	    if (i != classIndex) {
		m_InfoGains[i] = (ContingencyTables
			.entropyOverColumns(counts[i]) - ContingencyTables
			.entropyConditionedOnRows(counts[i]));
	    }
	}
	System.out.println("done building evaluator.");
    }

    /**
     * Reset options to their default values
     */
    protected void resetOptions() {
	m_InfoGains = null;
	m_missing_merge = true;
    }

    /**
     * evaluates an individual attribute by measuring the amount of information
     * gained about the class given the attribute.
     * 
     * @param attribute
     *            the index of the attribute to be evaluated
     * @return the info gain
     * @throws Exception
     *             if the attribute could not be evaluated
     */
    public double evaluateAttribute(int attribute) throws Exception {

	return m_InfoGains[attribute];
    }

    /**
     * Describe the attribute evaluator
     * 
     * @return a description of the attribute evaluator as a string
     */
    public String toString() {
	return "(binaryinfogainattributeeval)";
    }
}
