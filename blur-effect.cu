#include <stdio.h>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <stdlib.h>
#include <cuda_runtime.h>

using namespace std;
using namespace cv;

__global__ void blur(unsigned char *data, int width, int height, int channels, int kernel, int numThreads){
	int id = blockDim.x * blockIdx.x + threadIdx.x;
	int i = 0;
	int s= width * id / numThreads;
	int end =width * (id + 1) / numThreads;
	for( i =0 +s; i<end; i++){
		for (int j=0; j<height; j++){
			unsigned int blue=0.0, red=0.0, green=0.0;
			double sum = 0.0;
			for(int x=i-kernel; x<=i+kernel; x++){
				for(int y=j-kernel; y<=j+kernel; y++){
					if(x<width && x>=0 && y<height && y>=0){
						sum += 1;
						blue  += data[ (height*x*channels+y*channels)+0];
						green += data[ (height*x*channels+y*channels)+1];
						red   += data[ (height*x*channels+y*channels)+2];
					}
				}
			}
			data[ (height*i*channels+j*channels)+0] = (unsigned int) blue/sum;
			data[ (height*i*channels+j*channels)+1] = (unsigned int) green/sum;
			data[ (height*i*channels+j*channels)+2] = (unsigned int) red/sum;
		}
	}
}

int main(int args, char *argv[]){

	char *file = argv[1];
	char *file_output = argv[2];
	int kernel = atoi(argv[3]);
	int n_threads = atoi(argv[4]);
	cudaError_t err = cudaSuccess;
	Mat h_result, image;
	image = imread( file, CV_LOAD_IMAGE_UNCHANGED );
	h_result = image.clone();

	size_t imageSize = h_result.step[0] * h_result.rows;

	unsigned char *h_pixels = (unsigned char *) h_result.data;

	// Allocate the device input pixels
	unsigned char *d_pixels = NULL;
	err = cudaMalloc((void **)&d_pixels, imageSize);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device pixels (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// Copy the host input pixels in host memory to the device input vectors
    // in device memory
	err = cudaMemcpy(d_pixels, h_pixels, imageSize, cudaMemcpyHostToDevice);
	
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to copy pixels from host to device (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	
	// Calculate the number of blocks and the number of threads
	
	int threadsPerBlock = n_threads;
	int blocksPerGrid = imageSize / (threadsPerBlock*h_result.rows);
	int numThreads = threadsPerBlock * blocksPerGrid;
    
    // The function to parallelize
	
	blur<<<blocksPerGrid, threadsPerBlock>>>(d_pixels, h_result.rows, h_result.cols, h_result.channels(), kernel, numThreads);
	
	
	err = cudaGetLastError();
	if (err != cudaSuccess){
		cout<<"Failed to launch blur-effect at device"<< cudaGetErrorString(err)<<endl;
		exit(EXIT_FAILURE);
	}	
	
    // Copy the results of the device to the host
	err = cudaMemcpy(h_pixels, d_pixels, imageSize, cudaMemcpyDeviceToHost);
	
	if (err != cudaSuccess){
		cout<<"Failed at copy result image at host(error code )!"<< cudaGetErrorString(err)<<endl;
		exit(EXIT_FAILURE);
	}
	
	h_result.data = (uchar *) h_pixels;


    // Freed memory cuda
		
	err = cudaFree(d_pixels);
	if (err != cudaSuccess){
		cout<<"Failed at free memory at device (pixels)"<<endl;
		exit(EXIT_FAILURE);
	}
	
	err = cudaDeviceReset();
	if (err != cudaSuccess){
		cout<<"Failed at reset device"<<endl;
		exit(EXIT_FAILURE);
	}
	
	imwrite( file_output, h_result );
	return 0;
}