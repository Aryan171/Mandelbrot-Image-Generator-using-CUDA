// a program to generate a mandelbrot set image
// to include the external include directories goto project > 
// properties > VC++ directories > additional include diretrories and then paste the path of the file

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>

#define _CRT_SECURE_NO_WARNINGS

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

using fval = float;

__global__ void createImage(unsigned char* img, int side, bool isGrey, fval maxitr, int channels, int imgLen) {
    int pixelCount = threadIdx.x + (blockDim.x * blockIdx.x);
    int px = pixelCount * channels;
   
    
    if (px >= imgLen) {
        return;
    }

    int cx = 0.75 * ((pixelCount % side) - 0.3 * side);
    int cy = 0.75 * ((pixelCount / side) - 0.5 * side);

    fval mcx = -(static_cast<fval>(cx) * 4) / side,
        mcy = -(static_cast<fval>(cy) * 4) / side,
        x = mcx,
        y = mcy,
        h, 
        n = 0.0;

    while (n < maxitr && abs(x + y) < 16) {
        h = x;
        x = (x * x) - (y * y) + mcx;
        y = 2 * h * y + mcy;
        n = n + 1;
    }

    fval brightness = powf(n / maxitr, 0.5);

    if (isGrey == 1) {
        img[px] = brightness * 255;
    }
    else {
        img[px] = brightness * 255;
        img[px + 1] = brightness * brightness * 255;
        img[px + 2] = 255 / brightness; // bad code warning!!!!
    }
}

int main()
{
    char* file = new char[1000];
    std::cout << "enter the path where the output folder should be made" << std::endl;
    scanf(" %[^\n]s", file);
    strcat(file, "\\mandelbrot.png");

    int side;
    printf("enter the height/width of the mandelbrot graph\n");
    scanf("%d", &side);

    int halfSide = side / 2;

    fval maxitr;
    printf("enter the maximum iterations\n");
    scanf("%f", &maxitr);

    int isGrey;
    printf("enter 1 for grey image and 0 for color image\n");
    scanf("%d", &isGrey);

    int channels = (isGrey == 1) ? 1 : 3; 
    int imgLen = side * side * channels;
    
    // GPU HELP
    //**************************
    unsigned char* cudaImg;

    cudaMalloc(&cudaImg, imgLen * sizeof(char));

    createImage <<< ((side * side) / 1024) + 1, 1024 >>> (cudaImg, side, isGrey, maxitr, channels, imgLen);
    cudaDeviceSynchronize();

    // allocating memory for the image
    unsigned char* img = (unsigned char*)malloc(imgLen * sizeof(char));

    cudaMemcpy(img, cudaImg, imgLen, cudaMemcpyDeviceToHost);
    //***************************
    
    printf("writing the image\n");

    // writing the image
    stbi_write_png(file, side, side, channels, img, side * channels);

    printf("image written\n");

    // releasing the image memory
    stbi_image_free(img);
    cudaFree(cudaImg);
}