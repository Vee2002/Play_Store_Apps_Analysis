from flask import Flask, request, render_template
import joblib
import numpy as np
from saved_model import best_model

app = Flask(__name__)

# Loading the model
print(best_model)