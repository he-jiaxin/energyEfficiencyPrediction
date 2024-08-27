
# ThermalLogic

**ThermalLogic** is a project that builds upon the deep learning model architecture proposed in the paper [**'Deep Floor Plan Recognition using a Multi-task Network with Room-boundary-Guided Attention'**](https://arxiv.org/abs/1908.11025). Originally implemented by [zlzeng/DeepFloorplan](https://github.com/zlzeng/DeepFloorplan) and adapted for newer versions of TensorFlow and Python, my work extends this model to focus specifically on assessing energy efficiency in architectural design.

This AI application evaluates energy efficiency from building blueprints, providing actionable recommendations to architects and engineers during the early stages of design. The project, conducted under the UCL IXN initiative, also incorporates IBM Watsonx for code generation and text-to-speech features, enhancing accessibility for visually impaired users.

Central to **ThermalLogic** are two models: a deep learning model for image segmentation, inspired by Zeng et al. (2019), and a machine learning model tailored to predict cooling and heating loads in residential buildings. The application connects with AutoCAD through a file-based web interface, allowing for seamless design analysis and feedback without requiring direct integration with AutoCAD, making it both flexible and easy to implement in existing workflows.

By evaluating architectural features like compactness, orientation, and glazing, the AI tool predicts energy efficiency, providing insights through metrics like MAE and R-squared, and visualizing results with heatmaps and coolmaps. These tools highlight areas for design improvement, guiding early-stage architectural decisions towards greater sustainability. **ThermalLogic** not only advances AI applications in sustainable architecture but also fosters collaboration between academia and industry, laying the groundwork for future innovations in this field.

<img src="resources/appUI.png" width="50%" style="margin-right: 50px;"><img src="resources/appResult.png" width="50%">

## Requirements

The model can be installed and run in various environments depending on your hardware and operating system. Below is a table summarizing the installation commands based on different scenarios:

| OS     | Hardware | Application        | Command                                                                 |
|--------|----------|--------------------|-------------------------------------------------------------------------|
| Ubuntu | CPU      | Model Development  | `pip install -e .[tfcpu,dev,testing,linting]`                           |
| Ubuntu | GPU      | Model Development  | `pip install -e .[tfgpu,dev,testing,linting]`                           |
| MacOS  | M1 Chip  | Model Development  | `pip install -e .[tfmacm1,dev,testing,linting]`                         |
| Ubuntu | GPU      | Model Deployment API | `pip install -e .[tfgpu,api]`                                           |
| Ubuntu | GPU      | Everything         | `pip install -e .[tfgpu,api,dev,testing,linting,game]`                  |
| Agnostic | ...    | Docker             | (to be updated)                                                         |
| Ubuntu | GPU      | Notebook           | `pip install -e .[tfgpu,jupyter]`                                       |
| Ubuntu | GPU      | Game               | `pip install -e .[tfgpu,game]`                                          |

## How to Run

### 1. Install Packages

Choose one of the following methods depending on your preference for environment management:

#### Option 1: Python Virtual Environment

\`\`\`bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip setuptools wheel
\`\`\`

#### Option 2: Conda Environment (Preferred)

\`\`\`bash
conda create -n venv python=3.8 cudatoolkit=10.1 cudnn=7.6.5
conda activate venv
\`\`\`
