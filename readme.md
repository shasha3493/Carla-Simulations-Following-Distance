## Setup

The Code is written in Python  3.7.9 . If you don't have Python installed you can find it [here](https://www.python.org/downloads/). If you are using a lower version of Python you can upgrade using the pip package, ensuring you have the latest version of pip

If you are using Windows, run the following commands in PowerShell

1. Clone the repository

    `git clone`
  
2. Change to the working directory

    `cd `
 
3. Create the virtual environment and activate

    `virtualenv -p python3 venv`
    
    `.\venv\Scripts\activate`
    
4. Install the required packages

    `pip install -r requirements.txt`
    
5. Install PyTorch

    `pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116`  

## Usage

Launch CARLA and start the Unreal Editor

1. Prediction result of AI model on carla simulation

    `"python test_adas.py`
  
2. Examples from distance calculation of carla

    `python record_adas.py `