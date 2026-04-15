# License Strategy

BRepAX uses Apache License 2.0 for all original code, with a transitive dependency on OCCT (LGPL 2.1) via cadquery-ocp-novtk.

## What This Means for Users

### Academic and Research Use

No restrictions. LGPL permits unrestricted use for research, papers, and collaboration.

### Personal and Open-Source Use

No restrictions. The dynamic linking exception of LGPL 2.1 applies; users can freely use BRepAX in their projects regardless of license.

### Commercial Internal Use

No restrictions. Companies can use BRepAX internally without LGPL obligations (internal tools are not "distribution").

### Commercial Redistribution (Closed-Source)

LGPL requires that users can replace the LGPL component. Since OCCT is dynamically linked via Python wheels, this is satisfied by default. Users can install a different version of cadquery-ocp-novtk without modifying BRepAX.

## Industry Precedent

CadQuery (Apache 2.0) and build123d use the identical structure: Apache 2.0 wrapper around LGPL OCCT. Both have wide adoption in industry and academia without license issues.
