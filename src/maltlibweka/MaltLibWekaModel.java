package maltlibweka;

import java.io.Serializable;
import java.util.ArrayList;

import org.maltparser.ml.lib.MaltFeatureNode;
import org.maltparser.ml.lib.MaltLibModel;

import weka.classifiers.Classifier;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

public class MaltLibWekaModel implements Serializable, MaltLibModel {
	private static final long serialVersionUID = 3423414687296070741L;
	private Classifier classifier;
	private int classUpperBound;

	/**
	 * Construct a MaltLibWekaModel with the specified weka classifier and
	 * maximum class number. (classification might return any number up to the
	 * upperbound - 1)
	 * 
	 * @param classifier
	 * @param classUpperBound
	 */
	public MaltLibWekaModel(Classifier classifier, int classUpperBound) {
		this.classifier = classifier;
		this.classUpperBound = classUpperBound;
	}

	/**
	 * We're given an array of MaltFeatureNode, and we have to come up with a
	 * classification. In order to do that, produce a weka instance.
	 */
	@Override
	public int[] predict(MaltFeatureNode[] x) {
		ArrayList<Attribute> attinfo = new ArrayList<Attribute>();

		for (int featnum = 0; featnum < x.length; featnum++) {
			// We've decided that names of features start at 1; the names come
			// from their column numbers in the .ins file that we read them from
			// during training.
			Attribute e = new Attribute("" + (featnum + 1));
			attinfo.add(e);
		}

		ArrayList<String> possibleClasses = new ArrayList<String>();
		for (int cn = 0; cn < classUpperBound; cn++) {
			possibleClasses.add("" + cn);
		}

		Attribute classAttribute = new Attribute("class", possibleClasses);
		attinfo.add(classAttribute);

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
		int[] out = new int[classUpperBound];
		for (int i = 0; i < classUpperBound; i++) {
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
	private static Instance featuresToInstance(MaltFeatureNode[] features) {
		DenseInstance out = new DenseInstance(features.length + 1);
		for (int i = 0; i < features.length; i++) {
			MaltFeatureNode mfn = features[i];
			out.setValue(i, mfn.getValue());
		}
		return out;
	}
}
