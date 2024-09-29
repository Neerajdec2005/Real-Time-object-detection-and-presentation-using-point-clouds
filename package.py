# Install the required packages
!pip install opencv-python-headless
!pip install torch torchvision
!pip install matplotlib plotly
!pip install transformers

# Import necessary libraries
from IPython.display import display, Javascript
from google.colab.output import eval_js
from base64 import b64decode
import cv2
import numpy as np
import torch
import plotly.graph_objects as go
from transformers import BertTokenizer, BertModel