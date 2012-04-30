package maltlibweka;

import java.io.BufferedOutputStream;
import java.io.BufferedReader;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.Map;

import org.maltparser.core.exception.MaltChainedException;
import org.maltparser.core.feature.FeatureVector;
import org.maltparser.core.feature.value.FeatureValue;
import org.maltparser.core.feature.value.SingleFeatureValue;
import org.maltparser.ml.lib.FeatureList;
import org.maltparser.ml.lib.Lib;
import org.maltparser.ml.lib.LibException;
import org.maltparser.ml.lib.MaltFeatureNode;
import org.maltparser.ml.lib.MaltLibModel;
import org.maltparser.parser.guide.instance.InstanceModel;
import org.maltparser.parser.history.action.SingleDecision;

import weka.attributeSelection.Ranker;
import weka.classifiers.Classifier;
import weka.classifiers.functions.Logistic;
import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

/**
 * Extends MaltParser with parameterizable Weka classifiers. Extends the same
 * class as LibLinear and LibSvm, so can be used in a similar way. Just calls
 * out to Weka classifiers instead.
 * 
 * @author Alex Rudnick
 */
public class LibWeka extends Lib {

    public static final String OOV = "OOV";

    /**
     * The largest number for a class.
     */
    private int classUpperBound = 0;

    private Map<String, FastVector> nominalMap = null;

    private ArrayList<HashMap<String, Integer>> save_attributeValueCounts;

    private Integer minValueCount = 5;

    public LibWeka(InstanceModel owner, Integer learnerMode)
	    throws MaltChainedException {
	super(owner, learnerMode, "libweka");
	if (learnerMode == CLASSIFY) {
	    try {
		ObjectInputStream input = new ObjectInputStream(
			getInputStreamFromConfigFileEntry(".moo"));
		try {
		    model = (MaltLibModel) input.readObject();
		} finally {
		    input.close();
		}
	    } catch (ClassNotFoundException e) {
		throw new LibException("Couldn't load the libweka model", e);
	    } catch (Exception e) {
		throw new LibException("Couldn't load the libweka model", e);
	    }
	}
    }

    @Override
    public boolean predict(FeatureVector featureVector, SingleDecision decision)
	    throws MaltChainedException {
	FeatureList featureList = new FeatureList();

	for (int i = 0; i < featureVector.size(); i++) {
	    FeatureValue fv = featureVector.getFeatureValue(i);
	    SingleFeatureValue sfv = (SingleFeatureValue) fv;

	    if (sfv.getValue() == 1) {
		featureList.add(new MaltFeatureNode(i, sfv.getIndexCode()));
	    } else if (sfv.getValue() == 0) {
		featureList.add(new MaltFeatureNode(i, -1));
	    } else {
		throw new MaltChainedException(
			"did not expect a value other than 1 or 0.");
	    }
	}
	decision.getKBestList().addList(model.predict(featureList.toArray()));
	return true;
    }

    protected void trainInternal(FeatureVector featureVector)
	    throws MaltChainedException {
	try {
	    owner.getGuide()
		    .getConfiguration()
		    .getConfigLogger()
		    .info("Creating LIBWEKA model " + getFile(".moo").getName()
			    + "\n");

	    Instances instances = buildWekaInstances(getInstanceInputStreamReader(".ins"));
	    System.out.println("got instances");

	    // Pull up an object of a type specified in the options.
	    // Possible classes are defined in appdata/options.xml.
	    @SuppressWarnings("unchecked")
	    Class<Classifier> classifierClass = (Class<Classifier>) owner
		    .getGuide().getConfiguration()
		    .getOptionValue("libweka", "classifier");
	    Classifier classifier = classifierClass.newInstance();

	    FastVector attinfoBeforeHacks = getAttInfo(instances);
	    instances = classifierSpecificHacks(instances, classifier);
	    FastVector attinfoPostHacks = getAttInfo(instances);
	    System.out.println("specific hacks done");

	    String optstring = owner.getGuide().getConfiguration()
		    .getOptionValueString("libweka", "wekaopts");
	    String[] wekaopts = optstring.split(" ");
	    classifier.setOptions(wekaopts);

	    System.out.println("training classifier...");
	    System.out.println("num instances: " + instances.numInstances());
	    System.out.println("num attributes: " + instances.numAttributes());
	    classifier.buildClassifier(instances);
	    System.out.println("trained classifier.");
	    ObjectOutputStream output = new ObjectOutputStream(
		    new BufferedOutputStream(new FileOutputStream(getFile(
			    ".moo").getAbsolutePath())));
	    try {
		output.writeObject(new MaltLibWekaModel(classifier,
			attinfoBeforeHacks, attinfoPostHacks, getNominalMap(),
			getClassUpperBound()));
	    } finally {
		output.close();
	    }
	} catch (OutOfMemoryError e) {
	    throw new LibException(
		    "Out of memory. Please increase the Java heap size (-Xmx<size>). ",
		    e);
	} catch (IllegalArgumentException e) {
	    e.printStackTrace();
	    throw new LibException(
		    "The Weka learner was not able to redirect Standard Error stream. ",
		    e);
	} catch (SecurityException e) {
	    throw new LibException(
		    "The Weka learner cannot remove the instance file. ", e);
	} catch (IOException e) {
	    throw new LibException(
		    "The Weka learner cannot save the model file '"
			    + getFile(".mod").getAbsolutePath() + "'. ", e);
	} catch (Exception e) {
	    e.printStackTrace();
	    throw new LibException("The Weka learner broke in some other way",
		    e);
	}
    }

    /**
     * We need a list of the Attributes for a given weka dataset.
     * 
     * @param instances
     * @return
     */
    private FastVector getAttInfo(Instances instances) {
	FastVector out = new FastVector();
	for (int i = 0; i < instances.numAttributes(); i++) {
	    out.addElement(instances.attribute(i));
	}
	return out;
    }

    /**
     * Modifies the weka dataset based on specific knowledge we have about the
     * classifier. This is kind of terrible, but we're going to do it anyway.
     * 
     * @param instances
     * @param classifier
     * @return
     * @throws Exception
     */
    private Instances classifierSpecificHacks(Instances instances,
	    Classifier classifier) throws Exception {

	if (classifier instanceof Logistic) {
	    return binarizeAndFilter(instances);
	}

	return instances;
    }

    private Instances binarizeAndFilter(Instances instances) throws Exception {
	System.out.println("num attributes initially: "
		+ instances.numAttributes());
	try {
	    instances = FastBinarizer.fastBinarize(instances);
	} catch (OutOfMemoryError e) {
	    e.printStackTrace();
	    System.exit(1);
	}
	System.out.println("num attributes after binarization: "
		+ instances.numAttributes());
	instances = simpleFilterAttributes(instances, 200);
	System.out.println("num attributes at the end: "
		+ instances.numAttributes());
	return instances;
    }

    private Instances simpleFilterAttributes(Instances instances,
	    int maxAttributes) throws Exception {
	if (maxAttributes >= instances.numAttributes()) {
	    return new Instances(instances);
	}
	Ranker ranker = new Ranker();
	ranker.setNumToSelect(maxAttributes);
	BinaryInfoGainAttributeEval bigae = new BinaryInfoGainAttributeEval();
	bigae.buildEvaluator(instances);
	// GainRatioAttributeEval grae = new GainRatioAttributeEval();
	// grae.buildEvaluator(instances);
	// ChiSquaredAttributeEval chisquared = new ChiSquaredAttributeEval();
	// chisquared.buildEvaluator(instances);
	System.out.println("built the evaluator.");
	int[] ranked = ranker.search(bigae, instances);
	System.out.println("successfully ranked.");
	StringBuilder sb = new StringBuilder();
	for (int i = 0; i < maxAttributes; i++) {
	    if (i != 0) {
		sb.append(",");
	    }
	    // Remove uses 1-indexes like a champ.
	    sb.append(ranked[i] + 1);
	}
	sb.append("," + (instances.classIndex() + 1));

	Remove remove = new Remove();
	remove.setAttributeIndices(sb.toString());
	remove.setInvertSelection(true);
	remove.setInputFormat(instances);
	System.out.println("not deleting: " + sb);

	Instances out = Filter.useFilter(instances, remove);
	System.out.println("after filtering, num attributes: "
		+ out.numAttributes());
	return out;
    }

    private Instances buildWekaInstances(InputStreamReader isr)
	    throws MaltChainedException {
	try {
	    FastVector attinfo = new FastVector();
	    final BufferedReader fp = new BufferedReader(isr);
	    Instances out = null;
	    int nWekaFeatures = -1;

	    ArrayList<HashMap<String, Integer>> attributeValueCounts = getAttributeValueCounts();
	    Map<String, FastVector> themap = getNominalMap();

	    for (int featnum = 1; featnum < themap.keySet().size(); featnum++) {
		String featname = "" + featnum;
		// All attributes here are nominal now.
		Attribute e = new Attribute(featname, themap.get(featname));
		attinfo.addElement(e);

		// we have to fill in the featureMap.
		// XXX(alexr): be able to explain this...
		featureMap.addIndex(featnum, featnum - 1);
	    }
	    FastVector possibleClasses = new FastVector();
	    for (int cn = 0; cn < getClassUpperBound(); cn++) {
		possibleClasses.addElement("" + cn);
	    }
	    Attribute classAttribute = new Attribute("CLASS", possibleClasses);
	    attinfo.addElement(classAttribute);

	    out = new Instances("MaltFeatures", attinfo, 0);
	    out.setClass(classAttribute);
	    nWekaFeatures = attinfo.size();
	    while (true) {
		String line = fp.readLine();
		if (line == null)
		    break;

		String[] columns = tabPattern.split(line);
		String instanceClass = columns[0];

		// fill in the weka instance to add into the training data.
		Instance instance = new Instance(nWekaFeatures);
		instance.setDataset(out);
		for (int featnum = 1; featnum < nWekaFeatures; featnum++) {
		    Attribute att = (Attribute) attinfo.elementAt(featnum - 1);

		    String val = columns[featnum];
		    if (attributeValueCounts.get(featnum - 1).get(val) >= minValueCount) {
			instance.setValue(att, val);
		    } else {
			instance.setValue(att, OOV);
		    }
		}
		instance.setClassValue(instanceClass);
		out.add(instance);
	    }
	    return out;
	} catch (IOException ioe) {
	    throw new MaltChainedException("io exception", ioe);
	}
    }

    protected void trainExternal(FeatureVector featureVector)
	    throws MaltChainedException {
	throw new MaltChainedException("XXX trainExternal XXX oh hecks yeah");
    }

    public void terminate() throws MaltChainedException {
	super.terminate();
    }

    public void initLibOptions() {
	libOptions = new LinkedHashMap<String, String>();
    }

    /**
     * Find the highest class number in the instances file and add 1 to it, so
     * we know how many different possible classes there are for classification.
     * 
     * @return the highest class number, +1
     * @throws MaltChainedException
     */
    public int getClassUpperBound() throws MaltChainedException {
	if (classUpperBound != 0) {
	    return classUpperBound;
	}
	BufferedReader reader = new BufferedReader(
		getInstanceInputStreamReader(".ins"));
	try {
	    while (true) {
		String line = reader.readLine();
		if (line == null)
		    break;

		String[] columns = tabPattern.split(line);
		int instanceClass = Integer.parseInt(columns[0]);

		if ((1 + instanceClass) > classUpperBound) {
		    classUpperBound = (1 + instanceClass);
		}
	    }
	    reader.close();
	} catch (IOException e) {
	    throw new MaltChainedException("No instances found in file", e);
	}
	return classUpperBound;
    }

    public void initAllowedLibOptionFlags() {
	allowedLibOptionFlags = "";
    }

    /**
     * Build and return the map from names of features (which are just named by
     * their column from the file they're read from), to a FastVector of the
     * possible values that they can take.
     * 
     * @return
     * @throws MaltChainedException
     */
    public Map<String, FastVector> getNominalMap() throws MaltChainedException {
	if (nominalMap != null) {
	    return nominalMap;
	}
	ArrayList<HashMap<String, Integer>> attributeValueCounts = getAttributeValueCounts();
	BufferedReader reader = new BufferedReader(
		getInstanceInputStreamReader(".ins"));
	HashMap<String, HashSet<String>> featuresToValues = new HashMap<String, HashSet<String>>();
	try {
	    while (true) {
		String line = reader.readLine();
		if (line == null)
		    break;
		String[] columns = tabPattern.split(line);
		for (int featnum = 0; featnum < columns.length; featnum++) {
		    String featname = (featnum == 0) ? "CLASS" : "" + featnum;
		    String val = columns[featnum];
		    if (!featuresToValues.containsKey(featname)) {
			featuresToValues.put(featname, new HashSet<String>());
			if (featnum != 0) {
			    featuresToValues.get(featname).add(OOV);
			}
		    }
		    if (featnum == 0
			    || attributeValueCounts.get(featnum - 1).get(val) >= minValueCount) {
			featuresToValues.get(featname).add(val);
		    }
		}
	    }
	    reader.close();
	} catch (IOException e) {
	    throw new MaltChainedException("No instances found in file", e);
	}
	HashMap<String, FastVector> out = new HashMap<String, FastVector>();
	for (String featname : featuresToValues.keySet()) {
	    HashSet<String> values = featuresToValues.get(featname);
	    FastVector vec = new FastVector();
	    for (String val : values) {
		vec.addElement(val);
	    }
	    out.put(featname, vec);
	}
	nominalMap = out;
	return out;
    }

    /**
     * Accessor for the attribute-value counts; computes them on demand if
     * needed.
     * 
     * @return An ArrayList where each element is a Map mapping from a given
     *         possible value to the count of times it occurred in the training
     *         data. Positions correspond to the positions of the attribute.
     * @throws MaltChainedException
     */
    public ArrayList<HashMap<String, Integer>> getAttributeValueCounts()
	    throws MaltChainedException {
	if (save_attributeValueCounts == null) {
	    save_attributeValueCounts = countAttributeValues();
	}
	return save_attributeValueCounts;
    }

    /**
     * Produce the list of maps, where each map contains the counts for each
     * value, for the attribute in the corresponding position.
     * 
     * @return
     * @throws MaltChainedException
     */
    private ArrayList<HashMap<String, Integer>> countAttributeValues()
	    throws MaltChainedException {
	// weka with its magic non-parametrized types.
	ArrayList<HashMap<String, Integer>> out = new ArrayList<HashMap<String, Integer>>();
	int numAttributes = 0;

	try {
	    BufferedReader reader = new BufferedReader(
		    getInstanceInputStreamReader(".ins"));
	    while (true) {
		String line = reader.readLine();
		if (line == null)
		    break;
		String[] columns = tabPattern.split(line);

		if (out.size() == 0) {
		    numAttributes = columns.length;
		    // populate the output array.
		    for (int i = 0; i < numAttributes; i++) {
			out.add(new HashMap<String, Integer>());
		    }
		}

		for (int featnum = 1; featnum < columns.length; featnum++) {
		    HashMap<String, Integer> valueCounts = out.get(featnum - 1);
		    String val = columns[featnum];
		    if (!valueCounts.containsKey(val)) {
			valueCounts.put(val, 0);
		    }
		    valueCounts.put(val, valueCounts.get(val) + 1);
		}
	    }
	    reader.close();
	} catch (IOException e) {
	    throw new MaltChainedException("No instances found in file", e);
	}
	return out;
    }
}
