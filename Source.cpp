/**
* CS-315 (Distributed Scalable Computing) 
* Final Project: Edge Detection with Sobel Operator
* Group members:
* Minh Nguyen - mnguyen24@my.whitworth.edu
* Luke Wing - lwing23@my.whitworth.edu
*/
#include <iostream>
#include <iomanip>
#include <string>
#include <algorithm>
#include <cmath>
#include <math.h>

using namespace std;

#include "bmp.h"

#define NUM_IMAGES	5
# define M_PI  3.14159265358979323846  /* pi */
void mainfunc(RGB* pixel, int height, int width);
void d_convert_greyscale(RGB* pixels, int height, int width);


void convert_greyscale(RGB* pixels, int height, int width)

{
	for (int y = 0; y < height; ++y) { // for each row in image
		for (int x = 0; x < width; ++x) { // for each pixel in the row
			int index = y * width + x; // compute index position of (y,x) coordinate

			int grey = 0.3 * pixels[index].red + 0.59 * pixels[index].green + 0 * 11 * pixels[index].blue; // calculate grey scale

			pixels[index].red = min(grey, 255);
			pixels[index].green = min(grey, 255);
			pixels[index].blue = min(grey, 255);
		}
	}
}

void calculateWeight(int dim, double kernel[][11]) { 
	double sigma = 3.0; //standard deviation
	double r, s = 2.0 * sigma * sigma;
	int radius = dim / 2;
	double sum = 0.0;

	for (int x = -radius; x <= radius; x++) {
		for (int y = -radius; y <= radius; y++) {
			r = sqrt(x * x + y * y);
			kernel[x+radius][y+radius] = (exp(-(r * r) / s)) / (M_PI * s);
			sum += kernel[x + radius][y + radius];
		} 
	}

	for (int i = 0; i < dim; i++) {  //normalize the kernel
		for (int j = 0; j < dim; j++) {
			kernel[i][j] /= sum;
		}
	}
}

void gaussian_blur(RGB* pixels, int height, int width) {
	const int dimension = 11; // has to be odd
	int radius = dimension / 2;
	double GKernel[dimension][dimension];
	calculateWeight(dimension, GKernel);
	for (int y = radius; y < height - radius; ++y) {
		for (int x = radius; x < width - radius; ++x) {
			int index = y * width + x;

			int above_index = (y - 1) * width + x;
			int below_index = (y + 1) * width + x;
			int left_index = index - 1;
			int right_index = index + 1;
			float red_value = 0.0;
			float green_value = 0.0;
			float blue_value = 0.0;

			for (int kernelX = 0; kernelX < dimension; kernelX++) { // Iterate over the stencil and calculate average based on weight
				for (int kernelY = 0; kernelY < dimension; kernelY++) {
					int index_for_kernel = (y + kernelX - radius) * width + (x + kernelX - radius);
					
					auto kernel_value = GKernel[kernelX][kernelY];
					red_value += pixels[index_for_kernel].red * kernel_value;
					green_value += pixels[index_for_kernel].green * kernel_value;
					blue_value += pixels[index_for_kernel].blue * kernel_value;
				}
			}

			pixels[index].red = red_value;
			pixels[index].green = green_value;
			pixels[index].blue = blue_value;
		}
	}
}

/**
*  Computes the average of the red, green, and blue components of an image
*
* @param pixels  The array of RGB (Red, Green, Blue) components of each pixel in the image
* @param height  The height of the image
* @param width   The width of the image
*/
void compute_component_average(RGB* pixels, int height, int width)
{
	double total_red = 0, total_green = 0, total_blue = 0;

	for (int y = 0; y < height; ++y) { // for each row in image
		for (int x = 0; x < width; ++x) { // for each pixel in the row
			int index = y * width + x; // compute index position of (y,x) coordinate
			total_red += pixels[index].red; // add the red value at this pixel to total
			total_green += pixels[index].green; // add the green value at this pixel to total
			total_blue += pixels[index].blue; // add the blue value at this pixel to total
		}
	}

	cout << "Red average: " << total_red / (height * width) << endl;
	cout << "Green average: " << total_green / (height * width) << endl;
	cout << "Blue average: " << total_blue / (height * width) << endl;

}

int main()
{
	do {
		string image_archive[NUM_IMAGES] = { "lena.bmp", "marbles.bmp", "sierra_02.bmp", "snail.bmp", "field.bmp"};
		cout << "Select an image: \n";
		for (int i = 0; i < NUM_IMAGES; ++i)
			cout << i << ": " << image_archive[i] << endl;
		cout << NUM_IMAGES << ": exit\n";

		int choice;
		do {
			cout << "Please choice: ";
			cin >> choice;
			if (choice == NUM_IMAGES) {
				cout << "Goodbye!\n";
				exit(0);
			}
		} while (choice < 0 || choice > NUM_IMAGES);

		BitMap image(image_archive[choice]); // Load the bitmap image into the BitMap object

		// Display some of the image's properties
		cout << "Image properties\n";
		cout << setw(15) << left << "Dimensions: " << image.getHeight() << " by " << image.getWidth() << endl;
		cout << setw(15) << left << "Size: " << image.getImageSize() << " bytes\n";
		cout << setw(15) << left << "Bit encoding: " << image.getBitCount() << " bits\n\n";

		RGB* pixels = image.getRGBImageArray(); // get the image array of RGB (Red, Green, and Blue) components
		int height = image.getHeight();
		int width = image.getWidth();
		clock_t cpu_startTime, cpu_endTime;
		double cpu_ElapseTime = 0;
		// Blurring the image
		cpu_startTime = clock();
		cpu_endTime = clock();
		mainfunc(pixels, height, width);
		//d_convert_greyscale(pixels, height, width);

		int* gray_array = new int[height * width];
		for (int i = 0; i < height * width; i++) {
			gray_array[i] = 4;
		}
		// Assign the modified pixels back to the image
		image.setImageFromRGB(pixels);

		// Save this image in "test.bmp"
		image.saveBMP("result.bmp");

		cout << "Check out test.bmp (click on it) to see image processing result\n\n";
		char response = 'y';
		cout << "Do you wish to repeat? [y/n] ";
		cin >> response;
		if (response != 'y') {
			cout << "Sorry to see you go ...\n";
			exit(0);
		}
	} while (true);
}