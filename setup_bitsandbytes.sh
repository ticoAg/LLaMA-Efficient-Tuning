git clone https://gitee.com/ticoAg/bitsandbytes.git
cd bitsandbytes
CUDA_VERSION=117 make cuda11x
python setup.py install
cd ..
rm -rf bitsandbytes