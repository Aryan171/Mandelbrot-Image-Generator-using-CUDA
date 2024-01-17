// a program which swaps the channels of an image

// to include the external include directories goto project > 
// properties > VC++ directories > additional include diretrories and then paste the path of the file

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <string.h>

#define _CRT_SECURE_NO_WARNINGS

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

__global__ void imgFunction(unsigned char* img, unsigned char* outimg, int imgChannels, int outimgChannels, int pixelCount) {
	int px = threadIdx.x + (blockIdx.x * blockDim.x);
	if (px >= pixelCount) {
		return;
	}

	int a = px * imgChannels;
	int b = px * outimgChannels;

	outimg[b] = img[a + 1];
	outimg[b + 1] = img[a + 2];
	outimg[b + 2] = img[a];
}

int main()
{
	int width, height, channels;

	char fileName[1000],
		* outFile = "C:\\Users\\aryan\\OneDrive\\Desktop\\outputfile.png";
	
	printf("enter the path of the image file\n");
	scanf("%s[^\n]s", fileName);
	

	unsigned char* img = stbi_load(fileName, &width, &height, &channels, 0), // the input file that will be turned into black and white
		* outimg = (unsigned char*)malloc(width * height * 3 * sizeof(char));

	if (img == NULL) {
		printf("failed to load the image");
		return 1;
	}
	
	int pixelCount = width * height;

	unsigned char* cudaImg;
	unsigned char* cudaOutimg;

	cudaMalloc(&cudaImg, pixelCount * channels);
	cudaMalloc(&cudaOutimg, pixelCount * 3);

	cudaMemcpy(cudaImg, img, pixelCount * channels, cudaMemcpyHostToDevice);

	imgFunction <<< ((width * height) / 1024) + 1, 1024 >>> (cudaImg, cudaOutimg, channels, 3, pixelCount);
	cudaDeviceSynchronize();

	cudaMemcpy(outimg, cudaOutimg, pixelCount * 3, cudaMemcpyDeviceToHost);

	// writing the outimg
	stbi_write_png(outFile, width, height, 3, outimg, width * 3);
	// freeing the memory of the two images
	cudaFree(cudaImg);
	cudaFree(cudaOutimg);
	stbi_image_free(img);
	stbi_image_free(outimg);

	return 0;
}