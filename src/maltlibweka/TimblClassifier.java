package maltlibweka;

import java.io.BufferedWriter;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStreamWriter;
import java.net.ConnectException;
import java.net.SocketException;

import org.apache.commons.net.telnet.TelnetClient;

import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;

public class TimblClassifier extends Classifier {
  private static final long serialVersionUID = -1822137392036950026L;
  private TelnetClient telnetClient;

  public TimblClassifier() {
    telnetClient = null;
  }

  @Override
  public void buildClassifier(Instances data) throws Exception {
    int numAttributes = data.numAttributes();
    BufferedWriter out =
        new BufferedWriter(new OutputStreamWriter(new FileOutputStream("/tmp/thedataset"), "UTF-8"));

    for (int i = 0; i < data.numInstances(); i++) {
      StringBuffer row = new StringBuffer();
      Instance inst = data.instance(i);
      for (int attnum = 0; attnum < numAttributes - 1; attnum++) {
        row.append("f" + inst.value(attnum) + ",");
      }
      row.append("c" + inst.classValue() + "\n");
      out.write(row.toString());
    }
    out.close();
  }

  @Override
  public double classifyInstance(Instance instance) throws Exception {
    if (telnetClient == null) {
      initializeConnection();
    }

    BufferedWriter out = new BufferedWriter(new OutputStreamWriter(telnetClient.getOutputStream()));
    InputStream in = telnetClient.getInputStream();

    StringBuffer toSend = new StringBuffer();
    toSend.append("c ");
    for (int attnum = 0; attnum < instance.numAttributes() - 1; attnum++) {
      toSend.append("f" + instance.value(attnum) + ",");
    }
    toSend.append("?\n");

    String thereply = "";
    for (int tries = 0; !thereply.startsWith("CATEGORY {c") && tries < 10; tries++) {
      thereply = getResponseFor(toSend.toString(), out, in);
    }
    // We expect the reply to look like:
    // CATEGORY {cNUMBERGOESHERE}.
    String justthenumber = thereply.substring(thereply.indexOf('{') + 2, thereply.indexOf('}'));
    return Double.parseDouble(justthenumber);
  }

  @Override
  protected void finalize() throws Throwable {
    super.finalize();

    if (telnetClient != null) {
      try {
        BufferedWriter out =
            new BufferedWriter(new OutputStreamWriter(telnetClient.getOutputStream()));
        out.write("exit\n");
        out.flush();
      } finally {
        telnetClient.disconnect();
      }
    }
  }

  private String getResponseFor(String query, BufferedWriter out, InputStream in)
      throws IOException {
    StringBuffer reply = new StringBuffer();
    out.write(query);
    out.flush();
    while (true) {
      int c = in.read();
      if (c == -1 || (char) c == '\n') {
        break;
      }
      reply.append((char) c);
    }
    String thereply = reply.toString();
    return thereply;
  }

  private void initializeConnection() throws SocketException, IOException {
    telnetClient = new TelnetClient();
    for (int attempt = 0; attempt < 10; attempt++) {
      try {
        telnetClient.connect("localhost", 7000);
        // skip welcome.
        InputStream in = telnetClient.getInputStream();
        while (true) {
          int c = in.read();
          if (c == -1 || (char) c == '\n') {
            break;
          }
        }
      } catch (ConnectException e) {
        continue;
      }
      break;
    }
  }
}
