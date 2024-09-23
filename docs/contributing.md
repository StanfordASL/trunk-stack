# Contributing to the ASL Trunk robot project

Contributions are welcome! Here are some guidelines to follow when contributing to the project.

## Getting started
Start by cloning this repository using the following command:
```bash
gh repo clone hbuurmei/asl_trunk
```
where the GitHub CLI is required to use the `gh` command (I highly recommend it).

## Project layout
The project is organized as follows:
    
    asl_trunk/
        README.md  # The project README file.
        asl_trunk/  # The main package.
            asl_trunk_ws/  # The main ROS2 workspace, incl. data collection etc.
            mocap_ws/  # The ROS2 workspace for interfacing with the motion capture system.
            motor_control_ws/  # The ROS2 workspace for controlling the motors.
        docs/
            mkdocs.yml    # The website configuration file.
            docs/
                index.md  # The documentation homepage.
                contributing.md  # This file.
                ...       # Other markdown pages, images and other files.

## Code contributions
All the ROS2 packages are located in the `asl_trunk/` directory, and each workspace is their own repository.
These are added via git subtrees to have everything in one place.
Therefore, just contribute to the respective workspace repository, which will most likely be the [asl_trunk_ws](https://github.com/hbuurmei/asl_trunk_ws) repository.
Afterwards, the main repository can be updated with the new changes using the following command:
```bash
git subtree pull --prefix=asl_trunk/asl_trunk_ws https://github.com/hbuurmei/asl_trunk_ws.git main
```

## Contributing to the documentation
After cloning this repository, one can make updates to the documentation by editing the files in the `docs/` directory.
The documentation is built using [MkDocs](https://www.mkdocs.org/), a static site generator that's geared towards project documentation.
Specifically, we use the [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/) theme.
This should be installed using the following command:
```bash
pip install mkdocs-material
```
**Note:** The documentation is built automatically using GitHub Actions, so there is no need to build it locally. Always push to the `main` branch.
In case you want to preview the updates locally, simply use:
```bash
mkdocs serve
```
in the main directory, and open the browser as instructed.
