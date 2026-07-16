#!/bin/bash

uv pip install -r requirements.txt -extra-index-url  https://downloa
d.pytorch.org/whl/cpu

uv add --dev requirements.dev --extra-index-url  https://downloa
d.pytorch.org/whl/cpu