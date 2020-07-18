 #nvcc -o test  main.cpp decode_kernel.cu -arch=sm_75 -lcublas -lcudart 
 nvcc -o test  main.cpp  decoding.cu  decoder.cu -lcublas -lcudart -arch=sm_75 --std=c++11
