malt-libweka
============
Extensions to MaltParser so that you can plug in classifiers from Weka easily.

what's included
===============
* src: New code that I've written to extend MaltParser
* lib: libraries that we depend on
* config: config files required for malt-libweka

compiling and running
=====================
To build malt-libweka, use the included ant build.xml, like so:

    ant

You'll also have to modify your maltparser jar, but there is an included ant
task for this. Just say:

    ant updatemalt

licenses
========
malt-libweka depends on...

* libsvm.jar: see licenses/COPYRIGHT-libsvm
* log4j.jar: see licenses/LICENSE-APACHE2
* maltparser-1.7.1.jar: see licenses/LICENSE-MALTPARSER
* weka.jar: see licenses/COPYING-weka
