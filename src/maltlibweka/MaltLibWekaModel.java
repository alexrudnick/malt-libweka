package maltlibweka;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Map;

import org.maltparser.ml.lib.MaltFeatureNode;
import org.maltparser.ml.lib.MaltLibModel;

import weka.classifiers.Classifier;
import weka.classifiers.functions.Logistic;
import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.supervised.attribute.NominalToBinary;

public class MaltLibWekaModel implements Serializable, MaltLibModel {
    private static final long serialVersionUID = 3423414687296070741L;
    private Classifier classifier;
    private Map<String, FastVector> nominalMap;

    // remember the Instances object that we're using so we don't
    // have to construct it over and over again.
    private Instances save_instances = null;

    // attributes for weka instances, before feature binarization.
    private FastVector save_attinfo = null;

    // attributes for weka instances, what was actually used to train the
    // classifier.
    private ArrayList<Attribute> actualAttinfo;

    /**
     * Construct a MaltLibWekaModel with the specified weka classifier and
     * maximum class number. (classification might return any number up to the
     * upperbound - 1)
     * 
     * @param classifier
     * @param attinfo
     * @param map
     */
    public MaltLibWekaModel(Classifier classifier,
	    ArrayList<Attribute> attinfo, Map<String, FastVector> map) {
	this.classifier = classifier;
	this.nominalMap = map;
	this.actualAttinfo = attinfo;
    }

    /**
     * We're given an array of MaltFeatureNode, and we have to come up with a
     * classification. In order to do that, produce a weka instance.
     */
    @Override
    public int[] predict(MaltFeatureNode[] x) {
	double prediction = 0;
	try {
	    Instances instances = getInstances(x);
	    Instance instance = featuresToInstance(x, getAttInfo(x), instances);
	    instance.setDataset(instances);
	    instances.add(instance);

	    // We may need to mangle the attributes to match the attributes in
	    // the classifier.
	    instances = classifierSpecificHacks(instances);
	    prediction = classifier.classifyInstance(instances.firstInstance());
	} catch (Exception e) {
	    e.printStackTrace();
	    System.exit(1);
	}
	FastVector possibleClasses = nominalMap.get("CLASS");
	int[] out = new int[possibleClasses.size()];
	for (int i = 0; i < possibleClasses.size(); i++) {
	    out[i] = (int) prediction;
	}
	return out;
    }

    private Instances getInstances(MaltFeatureNode[] x) {
	// TODO Auto-generated method stub
	if (this.save_instances != null) {
	    return this.save_instances;
	}
	this.save_instances = new Instances("MaltFeatures", getAttInfo(x), 0);
	FastVector attinfo = getAttInfo(x);
	FastVector possibleClasses = nominalMap.get("CLASS");
	Attribute classAttribute = new Attribute("CLASS", possibleClasses);
	attinfo.addElement(classAttribute);
	save_instances.setClass(classAttribute);
	return this.save_instances;
    }

    private FastVector getAttInfo(MaltFeatureNode[] x) {
	if (this.save_attinfo != null) {
	    return this.save_attinfo;
	}
	
	this.save_attinfo = new FastVector();
	for (int featnum = 0; featnum < x.length; featnum++) {
	    // We've decided that names of features start at 1; the names come
	    // from their column numbers in the .ins file that we read them from
	    // during training.
	    String featname = "" + (featnum + 1);
	    Attribute e = new Attribute(featname, nominalMap.get(featname));
	    save_attinfo.addElement(e);
	}
	return this.save_attinfo;
    }

    private Instances classifierSpecificHacks(Instances instances)
	    throws Exception {
	if (classifier instanceof Logistic) {
	    return binarizeAndFilter(instances);
	}
	return instances;
    }

    /**
     * Build a weka Instance out of MaltFeatureNodes.
     * 
     * @param features
     *            Array of MaltFeatureNode.
     * @param instances
     * @param attinfo
     * @return
     */
    private Instance featuresToInstance(MaltFeatureNode[] features,
	    FastVector attinfo, Instances instances) {
	Instance out = new Instance(features.length + 1);
	out.setDataset(instances);
	for (int i = 0; i < features.length; i++) {
	    MaltFeatureNode mfn = features[i];
	    Attribute att = (Attribute) attinfo.elementAt(i);
	    String val = "" + Math.round(mfn.getValue());
	    if (!nominalMap.get(att.name()).contains(val)) {
		val = "OOV";
	    }
	    System.out.println("att: " + att);
	    System.out.println("val: " + val);
	    out.setValue(att, val);
	}
	out.setClassValue((String) nominalMap.get("CLASS").elementAt(0));
	return out;
    }

    /**
     * Change the nominal attributes into binary ones, then filter out the
     * attributes that weren't used to train the classifier.
     * 
     * @param instances
     * @return
     * @throws Exception
     */
    private Instances binarizeAndFilter(Instances instances) throws Exception {
	NominalToBinary nominalToBinary = new NominalToBinary();
	nominalToBinary.setInputFormat(instances);
	instances = Filter.useFilter(instances, nominalToBinary);
	for (int i = instances.numAttributes() - 1; i >= 0; i--) {
	    if (!actualAttinfo.contains(instances.attribute(i))) {
		if (instances.classAttribute() != instances.attribute(i)) {
		    instances.deleteAttributeAt(i);
		}
	    }
	}
	return instances;
    }

}
