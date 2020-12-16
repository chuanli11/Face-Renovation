# Install


```
virtualenv -p /usr/bin/python3.6 venv
. venv/bin/activate

pip install opencv-python
pip install tqdm
pip install imgaug
pip install torch==1.2.0 torchvision==0.4.0
pip install dill
pip install dominate

cd
git clone https://github.com/davisking/dlib.git
cd dlib
mkdir build; cd build; cmake ..; cmake --build .
cd ..
python setup.py install

cd
git clone https://github.com/Lotayou/Face-Renovation.git
cd Face-Renovation
```


# Data/Model Preparation

Download pretrained model from [here](https://yadi.sk/d/Pl_hxVZPa_PHew). Unzip it to the repo directory

Set the path to your testing in `degrade_lambda.py` and `config_hifacegan.py`.

```
python degrade_lambda.py
```

# Run

```
python test_lambda.py

```


