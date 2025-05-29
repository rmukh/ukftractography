UKF TRACTOGRAPHY
================

![Static Badge](https://img.shields.io/badge/RHEL-9.5-tamarind)
![Static Badge](https://img.shields.io/badge/GCC-11.5.0-yellow)
![Static Badge](https://img.shields.io/badge/Cmake-3.31.0-red)

![Static Badge](https://img.shields.io/badge/VTK-9.4.1-green)
![Static Badge](https://img.shields.io/badge/ITK-5.4.2-skyblue)
![Static Badge](https://img.shields.io/badge/Boost-1.87.0-magenta)
![Static Badge](https://img.shields.io/badge/Eigen-3.4.0-pink)

About
-----

We present a framework which uses an unscented Kalman filter for performing
tractography. At each point on the fiber the most consistent direction is found
as a mixture of previous estimates and of the local model.

It is very easy to expand the framework and to implement new fiber representations 
for it. Currently it is possible to tract fibers using two different 1-, 2-, or 3-tensor 
methods. Both methods use a mixture of Gaussian tensors. One limits the diffusion 
ellipsoids to a cylindrical shape (the second and third eigenvalue are assumed to be 
identical) and the other one uses a full tensor representation.

__Authors__:
Yogesh Rathi (yogesh@bwh.harvard.edu), Stefan Lienhard, Yinpeng Li, Martin
Styner, Ipek Oguz, Yundi Shi, Christian Baumgartner (c.f.baumgartner@gmail.com),
Ryan Eckbo, Tashrif Billah, Dheshan Mohandass

For references and further documentation, please see the [Slicer module homepage](https://www.slicer.org/slicerWiki/index.php/Documentation/Nightly/Modules/UKFTractography).

Installation
------------

### 1. Container

In April 2025, we revamped *ukftractography* installation infrastructure and made a Docker container to facilitate future
installation and usage. The container can be pulled from Docker Hub:

    docker pull tbillah/ukftractography:latest

Then, it can be used as:

    docker run --rm -v /local/path/to/data:/remote/data/ ukftractography \
    --dwiFile /remote/data/dwi.nhdr --maskFile /remote/data/mask.nhdr \
    --tracts /remote/data/tracts.vtk ...

As you see, the mandatory step is to mount local data into the Docker container first.
Then, you can reference the mounted data to construct ukftractography arguments e.g.
`--dwiFile`, `--maskFile`, etc.

However, to simply shell into the container, use:

    docker run --rm -ti --entrypoint /bin/bash ukftractography


### 2. From Source

Checkout from github:

    git clone https://github.com/pnlbwh/ukftractography.git

There are 3 ways to build this project from source. Way (a) is tested for every pull request to this repository.
Way (b) is rarely tested. Way (c) is done every night that can be checked at https://slicer.cdash.org/projects.


#### a) Standalone Superbuild

    cd <build-dir>
    cmake <path-to-source>
    make
    cd UKFTractography-build/UKFTractography
    ctest


#### b) Build with Slicer5

    cd <build-dir>
    cmake -DSlicer_DIR=<path-to-Slicer5-Superbuild>/Slicer-build <path-to-source>
    make
    cd UKFTractography-build/UKFTractography
    ctest


#### c) Build via Slicer ExtensionIndex build

Create local extension index following [these instructions](https://slicer.readthedocs.io/en/latest/developer_guide/extensions.html#extensions-build-system), containing at least `UKFTractography.s4ext` and `SlicerDMRI.s4ext` (required runtime dependency).

Notes:

* To manually test the `UKF Tractography` Slicer module, start Slicer using the launcher named `SlicerWithSlicerDMRI` available in `/path/to/SlicerDMRI-build/inner-build` directory. This ensure that the SlicerDMRI modules are loaded and that the required MRML diffusion nodes are registered (i.e vtkMRMLFiberBundleNode).

* It may be helpful to [test the exension upload](https://slicer.readthedocs.io/en/latest/developer_guide/extensions.html#build-test-package-and-upload-to-extensions-server) using your API key.

### 3. As a Slicer 5 Extension

Navigate to the Slicer Extension Manager and download `UKF Tractography` to
install it as a Slicer 5 module.


Basic Usage
-----------

### 1. As Command Line Module

The executable is called 'UKFTractography'. It can be found in:
    
    <build-dir>/UKFTractography-build/UKFTractography/bin/

In order to see all options run.

    ./UKFTractography --help 

In the source directory of the project you will find a shell script called [examples/sample_run.sh](examples/sample_run.sh).
It should give you an idea of what a function call could look like. 

Files `dataset_Mask.nrrd` and `seeds_full_cc.nrrd` in [Input](UKFTractography/Data/Input/) folder are mask and seed files of subject 100307
in hcp dataset, download the subject's preprocessed diffusion MRI data from https://db.humanconnectome.org/.


### 2. NIFTI support

`UKFTractography` CLI historically supports only NRRD format. In May 2025, we integrated [pnlNipype](https://github.com/pnlbwh/pnlNipype)
into it. Now you can use [ukf.py](https://github.com/pnlbwh/pnlNipype/blob/6543c163104b7ff0c686753c8721422138a70728/scripts/ukf.py)
on NIFTI files without any additional installation steps. It uses `UKFTractography` under the hood but converts
NIFTI to NRRD format on the fly. It can be found in:

    <build-dir>/pnlNipype/scripts/ukf.py --help


### 3. As Slicer 5 module

Navigate to the Slicer Extension Manager and download `UKF Tractography` to
install it as a Slicer 5 module.  There will be 3 modules under
`Diffusion-->Tractography`: `UKF Tractography`, `vtk2mask`, and `vtkFilter`.


Notes
-----

1. There are rounding errors that affect the accuracy of 2T FW tracts.
This explains why the [2T_FW ctest](https://github.com/pnlbwh/ukftractography/pull/169#issuecomment-2758623859) fails.

2. Several steps in the SuperBuild process download additional git repositories as CMake external projects. By default `UKFTractography_USE_GIT_PROTOCOL:BOOL=OFF` indicates to use `https://` instead of `git://`. This default should work behind most firewalls (the git protocol uses port 9418: if this is blocked by your firewall the build will fail). If download problems are encountered, please [file an issue](https://github.com/pnlbwh/ukftractography/issues/new).

3. In April 2025, there were several updates to *ukftractography* installation infrastructure. The updates
   were adequately tested on RHEL 9 OS. However, it was brought to our attention that, upon installing
   the current UKFTractography from Slicer Extension Manager in Apple Silicon Macs, Slicer-5 crashes when
   UKFTractography module is started. The solution is believed to be an OS X update: macOS 14.6.1 --> Sequoia 15.4.
   OS X is expected to ask the user to perform the update to fix this problem. So feel free to do the update.
