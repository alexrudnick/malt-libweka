package maltlibweka;

import java.io.BufferedInputStream;
import java.io.BufferedWriter;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
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

	@Override
	public void buildClassifier(Instances data) throws Exception {
		int numAttributes = data.numAttributes();
		BufferedWriter out = new BufferedWriter(new OutputStreamWriter(
				new FileOutputStream("/tmp/thedataset"), "UTF-8"));

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

	public TimblClassifier() {
		System.out.println("I GOT CONSTRUCTED!!");
		serverPort = -1;
		telnetClient = null;
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

	@Override
	protected void finalize() throws Throwable {
		// TODO Auto-generated method stub
		super.finalize();

		if (telnetClient != null) {
			try {
				BufferedWriter out = new BufferedWriter(new OutputStreamWriter(
						telnetClient.getOutputStream()));
				out.write("exit\n");
				out.flush();
			} finally {
				telnetClient.disconnect();
			}
		}
	}

	@Override
	public double classifyInstance(Instance instance) throws Exception {
		System.out.println("CLASSIFY INSTANCE");
		if (serverPort == -1) {
			// have to spin up the server.
		}

		if (telnetClient == null) {
			initializeConnection();
		}

		BufferedWriter out = new BufferedWriter(new OutputStreamWriter(
				telnetClient.getOutputStream()));
		InputStream in = telnetClient.getInputStream();

		StringBuffer toSend = new StringBuffer();
		toSend.append("c ");
		for (int attnum = 0; attnum < instance.numAttributes() - 1; attnum++) {
			toSend.append("f" + instance.value(attnum) + ",");
		}
		toSend.append("?\n");

		String thereply = "";

		for (int tries = 0; !thereply.startsWith("CATEGORY {c") && tries < 10; tries++) {
			System.out.println("loop: " + tries);
			thereply = getResponseFor(toSend.toString(), out, in);
		}

		System.out.println("THE REPLY: " + thereply);

		// XXX we might sometimes drop bytes somehow and this is worrisome.
		// We expect the reply to look like:
		// CATEGORY {cNUMBERGOESHERE} , but sometimes some letters in CATEGORY
		// are missing somehow.
		String justthenumber = thereply.substring(thereply.indexOf('{') + 2,
				thereply.indexOf('}'));
		System.out.println("just the number: " + justthenumber);
		return Double.parseDouble(justthenumber);
	}

	private String getResponseFor(String query, BufferedWriter out,
			InputStream in) throws IOException {
		StringBuffer reply = new StringBuffer();
		out.write(query);
		System.out.println("classify this one: " + query);
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

	private int serverPort;
	// private HashMap<String, Double> classToWekaClass;
}
