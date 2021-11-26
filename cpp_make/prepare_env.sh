#!/usr/bin/sh

export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda/lib64"

home_dir="/search/odin/xx"
software="$home_dir/Softwares"
lowq="$home_dir/lowq"
mkdir -p $home_dir
mkdir -p $software
mkdir -p $lowq


## move files
## tensorrt
mv ~/TensorRT-6.0.1.5.tar $software
cd $software
tar -xvf TensorRT-6.0.1.5.tar

## cmake
mv ~/cmake-3.12.3.tar.gz $software
tar -zxvf cmake-3.12.3.tar.gz
cd cmake-3.12.3
./configure --prefix=$software
make -j12
make install

## install dependencies
mv ~/FreeImage3180.zip $software
cd $software
unzip FreeImage3180.zip
cd FreeImage
make -j12
make install

## lq src code
cd $lowq

mv ~/lq_adpad.tar .
tar -xvf lq_adpad.tar


## env
export PATH="$PATH:${software}/bin"

## begin build
cd lq_porn_nausea_rethead_adpad
cd TensorRTcpp
# modefy the cmakelist file
sed -i 's/lowq\/TensorRT-7.0.0.11/xx\/Softwares\/TensorRT-6.0.1.5/g' CMakeLists.txt

cd build

make clean
rm -rf CMake*
cmake ..
make -j12

if [ -f pornTRT.plan ]; then
	rm pornTRT.plan
fi
if [ -f serialized_model_mobilenetv1.plan ]; then
	rm serialized_model_mobilenetv1.plan
fi
if [ -f serialized_model_resnet34v1b.plan ]; then
	rm serialized_model_resnet34v1b.plan
fi
./trt_build

cd ${lowq}/lq_porn_nausea_service_rethead_adpad
./copy_bin_lib.sh

