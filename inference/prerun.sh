apt-get update && apt-get install -y libgl1
python -m pip install --upgrade pip
pip install openvino-dev[onnx,pytorch]==2022.3.0
