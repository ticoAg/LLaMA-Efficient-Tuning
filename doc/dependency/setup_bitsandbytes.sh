git clone $1
cd bitsandbytes
CUDA_VERSION=$2 make cuda11x
python setup.py install
cd ..
rm -rf bitsandbytes