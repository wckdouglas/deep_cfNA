pip3 install --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.8.0-cp34-cp34m-linux_x86_64.whl

pip install keras

#sequencing tools
git clone https://github.com/wckdouglas/sequencing_tools.git
cd sequencing_tools
pip install -r requirements.txt
python setup.py install --user
