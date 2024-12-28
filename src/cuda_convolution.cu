/* cuda_convolution.cu */

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>
#include "lodepng.h"

#define CHECK_CUDA_ERROR(call)                                           \
    {                                                                    \
        const cudaError_t error = call;                                  \
        if (error != cudaSuccess) {                                      \
            std::cerr << "Error: " << __FILE__ << ", line " << __LINE__  \
                      << ": " << cudaGetErrorString(error) << std::endl; \
            exit(1);                                                     \
        }                                                                \
    }


const int KERNEL_SIZE = 3;
__constant__ float d_kernel[KERNEL_SIZE * KERNEL_SIZE]; /* Convolution kernel in constant memory */


/* CUDA kernel without shared memory */
__global__ void convolutionGlobal(const unsigned char * input, unsigned char * output, int width, int height) 
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) 
    {
        float sum = 0.0f;

        for (int ky = -1; ky <= 1; ++ky) 
        {
            for (int kx = -1; kx <= 1; ++kx) 
            {
                int ix = min(max(x + kx, 0), width - 1);
                int iy = min(max(y + ky, 0), height - 1);
                sum += input[iy * width + ix] * d_kernel[(ky + 1) * KERNEL_SIZE + (kx + 1)];
            }
        }
        output[y * width + x] = static_cast<unsigned char>(fminf(fmaxf(sum, 0.0f), 255.0f));
    }
}

/* CUDA kernel with shared memory */
__global__ void convolutionShared(const unsigned char * input, unsigned char * output, int width, int height) 
{
    extern __shared__ unsigned char s_input[];

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int lx = threadIdx.x + 1;
    int ly = threadIdx.y + 1;

    int local_width = blockDim.x + 2;

    /* Load input into shared memory */
    if (x < width && y < height) 
    {
        s_input[ly * local_width + lx] = input[y * width + x];
    }

    if (threadIdx.x == 0 && x > 0)
    {
        s_input[ly * local_width] = input[y * width + (x - 1)];
    }
    
    else if (threadIdx.x == blockDim.x - 1 && x < width - 1)
    {
        s_input[ly * local_width + lx + 1] = input[y * width + (x + 1)];
    }

    if (threadIdx.y == 0 && y > 0)
    {
        s_input[(ly - 1) * local_width + lx] = input[(y - 1) * width + x];
    }
    
    else if (threadIdx.y == blockDim.y - 1 && y < height - 1)
    {
        s_input[(ly + 1) * local_width + lx] = input[(y + 1) * width + x];
    }
    __syncthreads();

    /* Perform convolution */
    if (x < width && y < height) 
    {
        float sum = 0.0f;

        for (int ky = -1; ky <= 1; ++ky) 
        {
            for (int kx = -1; kx <= 1; ++kx) 
            {
                sum += s_input[(ly + ky) * local_width + (lx + kx)] * d_kernel[(ky + 1) * KERNEL_SIZE + (kx + 1)];
            }
        }
        output[y * width + x] = static_cast<unsigned char>(fminf(fmaxf(sum, 0.0f), 255.0f));
    }
}

/* Load image using lodepng */
std::vector<unsigned char> loadImage(const char * filename, unsigned int & width, unsigned int & height) 
{
    std::vector<unsigned char> image;
    unsigned error = lodepng::decode(image, width, height, filename);

    if (error) 
    {
        std::cerr << "Error loading image: " << lodepng_error_text(error) << std::endl;
        exit(1);
    }
    return image;
}

/* Save image using lodepng */
void saveImage(const char * filename, const std::vector<unsigned char> & image, unsigned int width, unsigned int height) 
{
    if (image.size() != width * height) 
    {
        std::cerr << "Error: Image size mismatch with dimensions: "
                  << "Expected " << (width * height) << " but got " << image.size() << std::endl;
        return;
    }
    std::vector<unsigned char> rgbaImage(width * height * 4, 255);

    for (size_t i = 0; i < width * height; ++i) 
    {
        rgbaImage[4 * i + 0] = image[i];
        rgbaImage[4 * i + 1] = image[i];
        rgbaImage[4 * i + 2] = image[i];
    }
    unsigned error = lodepng::encode(filename, rgbaImage, width, height);

    if (error) 
    {
        std::cerr << "Error saving image: " << lodepng_error_text(error) << std::endl;
        exit(1);
    }
}

void printSharedMemoryLimit() 
{
    cudaDeviceProp prop;
    int device;

    CHECK_CUDA_ERROR(cudaGetDevice(&device));
    CHECK_CUDA_ERROR(cudaGetDeviceProperties(&prop, device));

    std::cout << "Shared Memory Per Block: " << prop.sharedMemPerBlock << " bytes" << std::endl;
}

int main(int argc, char ** argv) 
{
    if (argc < 7) 
    {
        std::cerr << "Usage: " << argv[0] << " <input.png> <output.png> <mode (0=global, 1=shared)> <block_size> <grid_x> <grid_y>" << std::endl;
        return 1;
    }
    printSharedMemoryLimit();  /* Print shared memory limit for Python script */

    const char * inputFile  = argv[1];
    const char * outputFile = argv[2];

    int mode      = atoi(argv[3]);
    int blockSize = atoi(argv[4]);
    int gridX     = atoi(argv[5]);
    int gridY     = atoi(argv[6]);

    unsigned int width, height;
    auto hostInput = loadImage(inputFile, width, height);

    if (hostInput.size() != width * height * 4) 
    {
        std::cerr << "Error: Input image dimensions do not match expected size." << std::endl;
        return 1;
    }

    /* Grayscale conversion */
    size_t imageSize = width * height;
    std::vector<unsigned char> grayscaleInput(imageSize);

    for (size_t i = 0; i < imageSize; ++i) 
    {
        unsigned char r = hostInput[4 * i + 0];
        unsigned char g = hostInput[4 * i + 1];
        unsigned char b = hostInput[4 * i + 2];
        grayscaleInput[i] = static_cast<unsigned char>(0.299f * r + 0.587f * g + 0.114f * b);
    }
    std::vector<unsigned char> hostOutput(imageSize);
    unsigned char *d_input, *d_output;

    CHECK_CUDA_ERROR(cudaMalloc(&d_input, imageSize));
    CHECK_CUDA_ERROR(cudaMalloc(&d_output, imageSize));

    float h_kernel[KERNEL_SIZE * KERNEL_SIZE] = 
    {
        -1, -1, -1,
        -1,  8, -1,
        -1, -1, -1
    };
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_kernel, h_kernel, sizeof(h_kernel)));
    CHECK_CUDA_ERROR(cudaMemcpy(d_input, grayscaleInput.data(), imageSize, cudaMemcpyHostToDevice));

    dim3 block(blockSize, blockSize);
    dim3 grid(gridX, gridY);

    if (mode == 0) 
    {
        convolutionGlobal<<<grid, block>>>(d_input, d_output, width, height);
    } 

    else if (mode == 1) 
    {
        size_t sharedMemSize = (block.x + 2) * (block.y + 2) * sizeof(unsigned char);
        convolutionShared<<<grid, block, sharedMemSize>>>(d_input, d_output, width, height);
    } 

    else 
    {
        std::cerr << "Invalid mode. Use 0 for global memory or 1 for shared memory." << std::endl;
        return 1;
    }
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    CHECK_CUDA_ERROR(cudaMemcpy(hostOutput.data(), d_output, imageSize, cudaMemcpyDeviceToHost));
    
    saveImage(outputFile, hostOutput, width, height);

    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}
