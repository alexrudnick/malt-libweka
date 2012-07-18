package maltlibweka;

import java.util.Enumeration;

import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SparseInstance;

/**
 * Utility class to binarize weka Instances where all of the attributes are nominal, without running
 * out of memory. Why is weka's NominalToBinary so memory-intensive?
 * 
 */
public class FastBinarizer {
  /**
   * Given a weka dataset with nominal attributes (and only nominal attributes, except possibly the
   * class attribute), produce a new weka dataset where all of the attributes are binary numeric
   * ones. The class attribute should remain unchanged. NB: this deletes all instances from the
   * incoming dataset.
   * 
   * @param instances
   * @return
   */
  @SuppressWarnings("unchecked")
  public static Instances fastBinarize(Instances instances) {
    System.out.println("FastBinarizer.");
    FastVector attinfo = new FastVector();

    Enumeration<Attribute> nominals = (Enumeration<Attribute>) instances.enumerateAttributes();
    while (nominals.hasMoreElements()) {
      Attribute nominal = (Attribute) nominals.nextElement();
      if (instances.classAttribute().equals(nominal)) {
        continue;
      }
      Enumeration<String> values = (Enumeration<String>) nominal.enumerateValues();
      while (values.hasMoreElements()) {
        String value = values.nextElement();
        Attribute binary = new Attribute(nominal.name() + "=" + value);
        attinfo.addElement(binary);
      }
    }
    attinfo.addElement(instances.classAttribute());
    System.out.println("FastBinarizer: built attinfo with num attributes: " + attinfo.size());
    Instances out = new Instances("MaltFeatures", attinfo, 0);
    out.setClassIndex(out.numAttributes() - 1);

    int instancesBuilt = 0;
    while (instances.numInstances() > 0) {
      Instance nomInstance = instances.lastInstance();
      SparseInstance sparseBinarized = new SparseInstance(attinfo.size());
      for (int i = 0; i < attinfo.size(); i++) {
        Attribute binAttribute = (Attribute) attinfo.elementAt(i);
        if (binAttribute.name().equals("CLASS")) continue;
        String[] splitted = binAttribute.name().split("=");
        // names of attributes start at 1.
        int field = Integer.parseInt(splitted[0]) - 1;
        String val = splitted[1];
        sparseBinarized.setValue(i, nomInstance.stringValue(field).equals(val) ? 1 : 0);
      }
      sparseBinarized.setDataset(out);
      sparseBinarized.setClassValue(nomInstance.classValue());
      instancesBuilt++;
      if (instancesBuilt % 5000 == 0) {
        System.out.println("built this many instances: " + instancesBuilt);
      }
      instances.delete(instances.numInstances() - 1);
      out.add(sparseBinarized);
    }
    System.out.println("FastBinarizer: built new instances.");
    return out;
  }
}
