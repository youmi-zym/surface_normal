#!/bin/bash
NUM_CPU=8

mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(($NUM_CPU))
cd ..
