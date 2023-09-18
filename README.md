generate_mesh.py
=========

### Usage

generate_mesh.py is a python script that integrates [Gmsh](https://gmsh.info/) library and parametrically generates a 3D mesh file (mesh.msh2) of a Nuclear Power Plant (NPP) on soil consisting of a Reactor building at the center surrounded by an Auxiliary building. The user can enable or disable the generation of either buildings. The buildings can be embedded or not embedded into the soil. Optionally, Seismic Resonant Metamaterials can be generated around the Auxiliary building. For more information refer to [generate_mesh.py](./generate_mesh.py)

****This script was created for the needs of the paper with title: Dynamic Structure-Soil-Structure Interaction for Nuclear Power Plants****

### Required software

- python

### Execution

To execute this script, simply open a terminal and type:

```bash
python generate_mesh.py
```

### Licence

[GNU General Public License v3.0](./COPYING)
