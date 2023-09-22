# NPP mesh generator


## Description

### ``` generate_mesh.py ```

[generate_mesh.py](./generate_mesh.py) is a python script that integrates [Gmsh](https://gmsh.info/) library and parametrically generates a 3D mesh file (mesh.msh2) of a Nuclear Power Plant (NPP) on soil consisting of a Reactor building at the center surrounded by an Auxiliary building. The user can enable or disable the generation of either buildings. The buildings can be embedded or not embedded into the soil. Optionally, Seismic Resonant Metamaterials can be generated around the Auxiliary building. For more information refer to [generate_mesh.py](./generate_mesh.py).

>[!IMPORTANT]
>This script was initially developed for the paper: 

<p align="center">
<strong>" Dynamic Structure-Soil-Structure Interaction for Nuclear Power Plants "</strong>
</p>

<p align="center">
  <img src="https://github.com/ConstantinosKanellopoulos/images_for_my_repo/blob/master/overview_and_cross_section_of_NPP_model.png">
</p>



## Required software

- [python](https://www.python.org/)



## How to run

To run [generate_mesh.py](./generate_mesh.py), simply open a terminal and type:

```bash
python generate_mesh.py
```



## How to cite

If you would like to cite [generate_mesh.py](./generate_mesh.py), please give reference to the corresponding journal paper :



## Licence

[GNU General Public License v3.0](./COPYING)



## Acknowledgement

This research was part of the project '[INSPIRE](https://itn-inspire.eu/)' funded by the European Union’s Horizon 2020 research and innovation programme under the Marie Skłodowska-Curie grant agreement no. 813424. Additional support from [ETH Zurich](https://ethz.ch/en.html) is gratefully acknowledged.

<!-- <img align="center" src="https://github.com/ConstantinosKanellopoulos/images_for_my_repo/blob/master/logos.png"> -->

<p align="center">
  <img src="https://github.com/ConstantinosKanellopoulos/images_for_my_repo/blob/master/logos.png">
</p>

[![DOI](https://zenodo.org/badge/693150991.svg)](https://zenodo.org/badge/latestdoi/693150991)