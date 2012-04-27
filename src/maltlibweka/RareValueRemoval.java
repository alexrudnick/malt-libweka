package maltlibweka;

import java.util.Enumeration;

import weka.core.Attribute;
import weka.core.Instances;

/**
 * Utility class for stripping rare nominal values out of a weka dataset.
 */
public class RareValueRemoval {

    public static Instances removeRareValues(Instances instances, int minCount,
	    String OOV) {
	// for each attribute in the instances...
	for (int attnum = 0; attnum < instances.numAttributes(); attnum++) {
	    Attribute att = instances.attribute(attnum);
	    Enumeration values = att.enumerateValues();
	    // for each value for that attribute
	    // count how many times it occurs.
	    // if it happens fewer than minCount times, set it to OOV. (make
	    // sure that OOV is a possible value for this Attribute).
	    // construct a new Instances and return it (do we need to do that?)
	}
	return null;
    }
}
