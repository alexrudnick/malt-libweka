package maltlibweka;

import java.io.Serializable;
import java.util.Map;

import org.maltparser.ml.lib.MaltFeatureNode;
import org.maltparser.ml.lib.MaltLibModel;

import weka.classifiers.Classifier;
import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;

public class MaltLibWekaModel implements Serializable, MaltLibModel {
    private static final long serialVersionUID = 3423414687296070741L;
    private Classifier classifier;
    private Map<String, FastVector> nominalMap;

    /**
     * Construct a MaltLibWekaModel with the specified weka classifier and
     * maximum class number. (classification might return any number up to the
     * upperbound - 1)
     * 
     * @param classifier
     * @param map
     */
    public MaltLibWekaModel(Classifier classifier, Map<String, FastVector> map) {
	this.classifier = classifier;
	this.nominalMap = map;
    }

    /**
     * We're given an array of MaltFeatureNode, and we have to come up with a
     * classification. In order to do that, produce a weka instance.
     */
    @Override
    public int[] predict(MaltFeatureNode[] x) {
	FastVector attinfo = new FastVector();

	for (int featnum = 0; featnum < x.length; featnum++) {
	    // We've decided that names of features start at 1; the names come
	    // from their column numbers in the .ins file that we read them from
	    // during training.
	    String featname = "" + (featnum + 1);
	    Attribute e = new Attribute(featname, nominalMap.get(featname));
	    attinfo.addElement(e);
	}
	FastVector possibleClasses = nominalMap.get("CLASS");
	Attribute classAttribute = new Attribute("CLASS", possibleClasses);
	attinfo.addElement(classAttribute);

	Instances instances = new Instances("MaltFeatures", attinfo, 0);
	instances.setClass(classAttribute);
	Instance instance = featuresToInstance(x, attinfo, instances);
	instance.setDataset(instances);

	double prediction = 0;
	try {
	    prediction = classifier.classifyInstance(instance);
	} catch (Exception e) {
	    e.printStackTrace();
	    System.exit(1);
	}
	int[] out = new int[possibleClasses.size()];
	for (int i = 0; i < possibleClasses.size(); i++) {
	    out[i] = (int) prediction;
	}
	return out;
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
	    out.setValue(att, val);
	}
	return out;
    }
}
