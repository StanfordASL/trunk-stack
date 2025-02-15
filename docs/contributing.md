# Contributing to the ASL Trunk Robot project

Contributions are welcome! Here are some guidelines to follow when contributing to the project.

## Getting started
Start by cloning this repository, e.g. using the following command:
```bash
gh repo clone StanfordASL/trunk-stack
```
where the GitHub CLI is required to use the `gh` command.

## Project layout
The project is organized as follows:
    
    trunk-stack/
        README.md  # The project README file.
        stack/  # The stack.
            camera/   # The ROS2 workspace running the camera.
            gripper/  # The ROS2 workspace enabling the gripper operation.
            main/     # The main ROS2 workspace, incl. data collection etc.
            mocap/    # The ROS2 workspace for interfacing with the motion capture system.
            motors/   # The ROS2 workspace for controlling the motors.
        docs/  # Documentation directory.
            docs/
                index.md         # The documentation homepage.
                contributing.md  # This file.
                ...              # Other markdown pages, images and other files.
        mkdocs.yml    # The website configuration file.

## Code Contributions
All the ROS2 packages are located within the `stack/` directory.
For simplicity, this is a monorepo, meaning that all the packages are in the same repository.
However, most likely contributions will be made to the `main/` package, where e.g. the controller and the main logic are located.

## Contributing to Documentation
After cloning this repository, one can make updates to the documentation by editing the files in the `docs/` directory.
The documentation is built using [MkDocs](https://www.mkdocs.org/), a static site generator that's geared towards project documentation.
Specifically, we use the [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/) theme.
This should be installed using the following command:
```bash
pip install mkdocs-material
```
Additionally, we make use of a number of plugins, that can be installed as:
```bash
pip install pymdown-extensions mkdocs-git-committers-plugin-2 mkdocs-git-revision-date-localized-plugin
```
**Note:** The documentation is built automatically using GitHub Actions, so there is no need to build it locally. Always push to the `main` branch.
In case you want to preview the updates locally, simply use:
```bash
mkdocs serve
```
in the main directory, and open the preview in the browser as instructed.
