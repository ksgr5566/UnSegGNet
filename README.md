# UnSegGNet
Official code for the paper UnSegGNet: Unsupervised Image Segmentation using Graph Neural Networks

# Steps to set up the repository

1. Clone and cd to this repo.
2. `python -m venv venv`
3. If Windows:
     `.\venv\Scripts\activate`
   <br/>
   If Mac:
     `. venv/bin/activate`
4. `pip install poetry==1.8.2`
5. `poetry install`\
    Note: If poetry couldn't install deeplake, please try manually `pip install deeplake==3.9.0`
6. Pip install the required torch version from [here](https://pytorch.org/). In this project we are using PyTorch 2.0 version.
7. Make the script.sh executable, run: `./script.sh`