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
	    Attribute e = new Attribute("" + (featnum + 1));
	    attinfo.addElement(e);
	}
	FastVector possibleClasses = nominalMap.get("CLASS");
	Attribute classAttribute = new Attribute("class", possibleClasses);
	attinfo.addElement(classAttribute);

	Instances instances = new Instances("MaltFeatures", attinfo, 0);
	instances.setClass(classAttribute);
	Instance instance = featuresToInstance(x);
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
     * @return
     */
    private Instance featuresToInstance(MaltFeatureNode[] features) {
	Instance out = new Instance(features.length + 1);
	for (int i = 0; i < features.length; i++) {
	    MaltFeatureNode mfn = features[i];
	    String featname = "" + (1 + i);
	    String val = "" + Math.round(mfn.getValue());
	    out.setValue(i, nominalMap.get(featname).contains(val) ? val : "OOV");
	}
	return out;
    }
}
