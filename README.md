# SterileBrowsingMedicalImgs
A gesture-based tool enabling sterile, hands-free manipulation of radiology images in the OR, enhancing surgical focus, reducing infection risk, and maintaining sterility. This scalable solution allows surgeons to access and navigate patient images through gestures, promoting efficiency and supporting a seamless surgical workflow.

### GPU dependencies (installed via conda):
`conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0absl-py==2.1.0`

### worflow steps:
Update config.yaml
Update secrets.yaml [Optional]
Update params.yaml
Update the entity
Update the configuration manager in src config
Update the components
Update the pipeline
Update the main.py
Update the dvc.yaml


### Create ECR Repo and get URI 
    - URI 529088256673.dkr.ecr.us-east-2.amazonaws.com/dlprojgestures


### Run and install docker on machine amzn linux 2023
```bash
sudo dnf update -y
sudo dnf install docker -y
sudo systemctl start docker
sudo systemctl enable docker
sudo systemctl status docker
sudo usermod -aG docker $USER
```