#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "RGB.h"
#include <stdio.h>
#include <vector>
#include <cmath>
#include <math.h>
#include <iostream>
# define M_PI  3.14159265358979323846  /* pi */

using namespace std;
/**
* Helper function to calculate the greyscale value based on R, G, and B
*/

__host__ int calcBlockDim(int total, int num_threads)
{
	int r = total / num_threads;
	if (total % num_threads != 0) // add one to cover all the threads per block
		++r;
	return r;
}


__host__ void calculateWeight(double*& kernel, int dim, int radius) {
	// fill the vector first
	double sigma = 1.0; //standard deviation
	double r, s = 2.0 * sigma * sigma;
	double sum = 0.0;

	for (int x = 0; x < dim; x++) {
		for (int y = 0; y < dim; y++) {
			int a = x - radius;
			int b = y - radius;
			r = sqrt(a * a + b * b);
			kernel[y * dim + x] = (exp(-(r * r) / s)) / (M_PI * s);
			sum += kernel[y * dim + x];
		}
	}

	for (int i = 0; i < dim * dim; i++) {  //normalize the kernel
		kernel[i] /= sum;
	}
	//change this to the exisiting formula in source

}

__global__ void Gaussian_filter(RGB* d_pixels, double* kernelKernel, const int inWidth,  const int inHeight, const int kernelDim) {

	int col = blockIdx.x * blockDim.x + threadIdx.x; // width
	int row = blockIdx.y * blockDim.y + threadIdx.y; // height
	int index = row * inWidth + col;


	RGB* pixel = d_pixels;
	double* GKernel = kernelKernel;
	__syncthreads();

	
	int radius = kernelDim / 2;
	double red_value = 0.0;
	double green_value = 0.0;
	double blue_value = 0.0;

	//this was changed from rgb to double
	
	//calculate the bounds

	// sum of all pixels in the area of the kernel weighted by the stencil's kernel's values
	if (col >= radius && row >= radius && col < inWidth - radius && row < inHeight - radius) {
		for (int i = 0; i < kernelDim * kernelDim; i++) {
			int index_for_kernel = index - (i - ((kernelDim * kernelDim) / 2));
			double kernel_value = GKernel[i];
			red_value += pixel[index_for_kernel].red* kernel_value;
			green_value += pixel[index_for_kernel].green * kernel_value;
			blue_value += pixel[index_for_kernel].blue * kernel_value;
		}
	}

	d_pixels[index].red = red_value;
	d_pixels[index].green = green_value;
	d_pixels[index].blue = blue_value;
	
}



// calculate the edge on the horizontal direction
__global__ void xGradient(RGB* d_pixels, double *x_gradient, int inHeight, int inWidth) {
	int col = blockIdx.x * blockDim.x + threadIdx.x; // width
	int row = blockIdx.y * blockDim.y + threadIdx.y; // height
	int index = row * inWidth + col;

	const int kernelDim = 3;
	int radius = kernelDim / 2;
	//x direction matrix
	int x_kernel[kernelDim * kernelDim] = { -1, 0, 1, -2, 0, 2, -1, 0, 1 };
	 
	if (col >= radius && row >= radius && col < inWidth - radius && row < inHeight - radius) {
		double sum = 0;
		int k_ind = 0;

		for (int i = 0; i < kernelDim * kernelDim; i++) {
			sum += x_kernel[i] * d_pixels[index - (i - ((kernelDim * kernelDim) / 2))].red; //sum of the sobel stencil
		}
		__syncthreads();
		x_gradient[index] = sum;

	}
}

//calculate the edge on the vertical direction
__global__ void yGradient(RGB* d_pixels, double* y_gradient, int inHeight, int inWidth) {
	int col = blockIdx.x * blockDim.x + threadIdx.x; // width
	int row = blockIdx.y * blockDim.y + threadIdx.y; // height
	int index = row * inWidth + col;

	const int kernelDim = 3;
	int radius = kernelDim / 2;
	// y direction matrix
	int y_kernel[kernelDim * kernelDim] = { -1, -2, -1, 0, 0, 0, 1, 2, 1};

	if (col >= radius && row >= radius && col < inWidth - radius && row < inHeight - radius) {
		double sum = 0;
		int k_ind = 0;
		
		for (int i = 0; i < kernelDim * kernelDim; i++) {
			sum += y_kernel[i] * d_pixels[index - (i - ((kernelDim * kernelDim) / 2))].red; //sum of the sobel stencil
		}
		__syncthreads();

		y_gradient[index] = sum;

	}
}

//combine points on x and y directions
__global__ void combine_xy(RGB* d_pixels, double* x_gradient, double* y_gradient,  int inHeight, int inWidth) {
	int col = blockIdx.x * blockDim.x + threadIdx.x; // width
	int row = blockIdx.y * blockDim.y + threadIdx.y; // height
	int index = row * inWidth + col;
	//get the gradient calculated for each pixel
	double x = x_gradient[index];
	double y = y_gradient[index];
	double G = sqrt((double)(x * x + y * y)); //G = sqrt(x^2 + y^2)
	
	if (G > 110) { //if G exceeds our threshold, set pixels to white, these are edge pixels
		d_pixels[index].red = 255;
		d_pixels[index].green = 255;
		d_pixels[index].blue = 255;

	}
	else { // if G does not exceed threshold, make them dark gray
		d_pixels[index].red = 50;
		d_pixels[index].green = 50;
		d_pixels[index].blue = 50;

	}


}

__device__ int greyscale(BYTE red, BYTE green, BYTE blue)
{
	int grey = 0.2126 * red + 0.7152 * green + 0.0722 * blue; // calculate grey scale
	return min(grey, 255);
}

/**
* Kernel for executing on GPY
*/
__global__ void greyscaleKernel(RGB* d_pixels, int height, int width)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x; // width
	int y = blockIdx.y * blockDim.y + threadIdx.y; // height

	if (y >= height || y >= width)
		return;

	int index = y * width + x;

	int grey = greyscale(d_pixels[index].red, d_pixels[index].green, d_pixels[index].blue); // calculate grey scale

	d_pixels[index].red = grey;
	d_pixels[index].green = grey;
	d_pixels[index].blue = grey;

}

/**
*	Host function for launching greyscale kernel
*/
__host__ void d_convert_greyscale(RGB* pixel, int height, int width)
{
	RGB* d_pixel;

	cudaMalloc(&d_pixel, height * width * sizeof(RGB));
	cudaMemcpy(d_pixel, pixel, height * width * sizeof(RGB), cudaMemcpyHostToDevice);

	dim3 grid, block;
	block.x = 16;
	block.y = 16;
	grid.x = calcBlockDim(width, block.x);
	grid.y = calcBlockDim(height, block.y);

	greyscaleKernel << <grid, block >> > (d_pixel, height, width);

	cudaMemcpy(pixel, d_pixel, height * width * sizeof(RGB), cudaMemcpyDeviceToHost);
}




__host__ void mainfunc(RGB* pixel, int height, int width) {
	cout << "Execute edge dectection kernels" << endl;
	const int kernelDim = 1;
	int radius = kernelDim / 2;

	int inCols = width;
	int inRows = height;
	int outCols = inCols - (kernelDim - 1);
	int outRows = inRows - (kernelDim - 1);
	RGB* d_pixels;
	double* kernelKernel;
	dim3 dimGrid, dimBlock;
	dimBlock.x = 16;
	dimBlock.y = 16;
	dimGrid.x = calcBlockDim(width, dimBlock.x);
	dimGrid.y = calcBlockDim(height, dimBlock.y);
	double* GKernel = new double[kernelDim * kernelDim];
	for (int i = 0; i < kernelDim * kernelDim; i++) {
		GKernel[i] = 0.0;
	}

	double* x_gradient;
	double* y_gradient;
	/*
	for (int i = 0; i < height * width; i++) {
		test_array[i] = 0;
	}*/
	//* x_gradient_p = 0;
	calculateWeight(GKernel, kernelDim, radius);
	cudaMalloc(&d_pixels, height * width * sizeof(RGB));
	cudaMalloc(&kernelKernel, kernelDim * kernelDim * sizeof(double));
	cudaMalloc(&x_gradient, height * width * sizeof(double) * 2.0); //extend memory for larger images
	cudaMalloc(&y_gradient, height * width * sizeof(double) * 2.0);


	cudaMemcpy(d_pixels, pixel, height * width * sizeof(RGB), cudaMemcpyHostToDevice);
	cudaMemcpy(kernelKernel, GKernel, kernelDim * kernelDim * sizeof(double), cudaMemcpyHostToDevice);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	float elapsed = 0;

	
	cudaEventRecord(start, 0);
	//1. Blur the image to reduce noise
	Gaussian_filter << <dimGrid, dimBlock >> > (d_pixels, kernelKernel, inCols, inRows, kernelDim);
	//2. Convert the image to gray scale to reduce processing resources, RGB has same value
	greyscaleKernel << < dimGrid, dimBlock >> > (d_pixels, height, width);
	cudaFree(kernelKernel);
	//3. Calculate edges on x and y directions
	xGradient << <dimGrid, dimBlock >> > (d_pixels, x_gradient, height, width);
	yGradient << <dimGrid, dimBlock >> > (d_pixels, y_gradient, height, width);

	//4. Combine the points from step 3
	combine_xy << <dimGrid, dimBlock >> > (d_pixels, x_gradient, y_gradient, height, width);
	cudaEventRecord(stop, 0);
	cudaDeviceSynchronize();
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsed, start, stop);
	printf("The elapsed time in gpu was %.2f ms\n", elapsed);


	cudaMemcpy(pixel, d_pixels, height * width * sizeof(RGB), cudaMemcpyDeviceToHost);
}