package maltlibweka;

import java.io.BufferedOutputStream;
import java.io.BufferedReader;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.Map;

import libsvm.svm_parameter;

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

import weka.classifiers.Classifier;
import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;

/**
 * Extends MaltParser with parameterizable Weka classifiers. Extends the same
 * class as LibLinear and LibSvm, so can be used in a similar way. Just calls
 * out to Weka classifiers instead.
 * 
 * @author Alex Rudnick
 */
public class LibWeka extends Lib {

    /**
     * The largest number for a class.
     */
    private int classUpperBound = 0;

    private Map<String, FastVector> nominalMap = null;

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
	    final Instances instances = buildWekaInstances(getInstanceInputStreamReader(".ins"));
	    owner.getGuide()
		    .getConfiguration()
		    .getConfigLogger()
		    .info("Creating LIBWEKA model " + getFile(".moo").getName()
			    + "\n");

	    // Pull up an object of a type specified in the options.
	    // Possible classes are defined in appdata/options.xml.
	    @SuppressWarnings("unchecked")
	    Class<Classifier> classifierClass = (Class<Classifier>) owner
		    .getGuide().getConfiguration()
		    .getOptionValue("libweka", "classifier");
	    Classifier classifier = classifierClass.newInstance();
	    // XXX(alexr): set classifier options here, get them from the
	    // command line.
	    // classifier.setOptions(null);
	    classifier.buildClassifier(instances);
	    ObjectOutputStream output = new ObjectOutputStream(
		    new BufferedOutputStream(new FileOutputStream(getFile(
			    ".moo").getAbsolutePath())));
	    try {
		output.writeObject(new MaltLibWekaModel(classifier,
			getNominalMap()));
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

    private Instances buildWekaInstances(InputStreamReader isr)
	    throws MaltChainedException {
	try {
	    FastVector attinfo = new FastVector();
	    final BufferedReader fp = new BufferedReader(isr);
	    Instances out = null;
	    int nWekaFeatures = -1;

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
	    while(true) {
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
		    instance.setValue(att, columns[featnum]);
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
	libOptions.put("s", Integer.toString(svm_parameter.C_SVC));
	libOptions.put("t", Integer.toString(svm_parameter.POLY));
	libOptions.put("d", Integer.toString(2));
	libOptions.put("g", Double.toString(0.2));
	libOptions.put("r", Double.toString(0));
	libOptions.put("n", Double.toString(0.5));
	libOptions.put("m", Integer.toString(100));
	libOptions.put("c", Double.toString(1));
	libOptions.put("e", Double.toString(1.0));
	libOptions.put("p", Double.toString(0.1));
	libOptions.put("h", Integer.toString(1));
	libOptions.put("b", Integer.toString(0));
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
	allowedLibOptionFlags = "stdgrnmcepb";
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
			featuresToValues.get(featname).add("OOV");
		    }
		    featuresToValues.get(featname).add(val);
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
}
