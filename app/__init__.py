from flask import Flask
from flask_cors import CORS
from flasgger import Swagger

app = Flask(__name__)
swagger = Swagger(app)
CORS(app, resources={r"/api/*": {"origins": "*"}})

from app import routes
