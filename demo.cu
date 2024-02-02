#include <stdio.h>
#include "lodepng.h"
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <fstream>
#include <cuda.h>
#include "Halide.h"
#include "halide_image_io.h"

#include <chrono>


#define STRINGIFY(x) #x
#define STR(x) STRINGIFY(x)
#define FILE_LINE __FILE__ ":" STR(__LINE__)
#define CUDA_CHECK_THROW(x)                                                                            \
do {                                                                                                   \
    cudaError_t result = x;                                                                            \
    if (result != cudaSuccess) {                                                                       \
        std::cout << FILE_LINE << " CUDA ERROR: " << cudaGetErrorString(result) << std::endl;          \
        exit(-1);                                                                                      \
    }                                                                                                  \
} while(0);

#define INPUT_FILENAME "/home/snowsr/projects/csc2231/halide_proj/img.png"
#define CUDA_OUTPUT_FILENAME "/home/snowsr/projects/csc2231/halide_proj/img2.png"
#define HALIDE_OUTPUT_FILENAME "/home/snowsr/projects/csc2231/halide_proj/img3.png"
#define KERNEL_SIZE 5

using namespace std::chrono;
using namespace Halide;
using namespace Halide::Tools;

__global__ void convolve_kernel(const unsigned char *input_img,
                                unsigned char *output_img, const unsigned width,
                                const unsigned height) {
    const unsigned idx = threadIdx.x + blockIdx.x * 1024;
    if (idx >= width * height)
        return;
    const long x = idx % width;
    const long y = idx / width;
    constexpr unsigned ext = KERNEL_SIZE / 2;
    int rv = 0, gv = 0, bv = 0;
    unsigned ct = 0;

    for (long yy = y - ext; yy <= y + ext; ++yy) {
        const long yi = min(max(yy, 0L), static_cast<long>(height) - 1);
        for (long xx = x - ext; xx <= x + ext; ++xx) {
            const long xi = min(max(xx, 0L), static_cast<long>(width) - 1);
            rv += static_cast<int>(input_img[(yi * width + xi) * 3]);
            gv += static_cast<int>(input_img[(yi * width + xi) * 3 + 1]);
            bv += static_cast<int>(input_img[(yi * width + xi) * 3 + 2]);
            ++ct;
        }
    }

    output_img[idx * 3] = static_cast<unsigned char>(
        max(static_cast<int>(input_img[idx * 3]) - rv / ct, 0));
    output_img[idx * 3 + 1] = static_cast<unsigned char>(
        max(static_cast<int>(input_img[idx * 3 + 1]) - gv / ct, 0));
    output_img[idx * 3 + 2] = static_cast<unsigned char>(
        max(static_cast<int>(input_img[idx * 3 + 2]) - bv / ct, 0));
}

void customCUDA() {
    unsigned char *h_in_img = nullptr;
    unsigned char *h_out_img;
    unsigned width, height;

    unsigned error = lodepng_decode24_file(&h_in_img, &width, &height,
                                           INPUT_FILENAME);
    if (error) {
        printf("error %u: %s\n", error, lodepng_error_text(error));
        exit(-1);
    }

    const auto t1 = duration_cast<milliseconds>(
        system_clock::now().time_since_epoch()).count();

    cudaStream_t stream;
    CUDA_CHECK_THROW(cudaStreamCreate(&stream));

    const size_t mem_size = height * width * sizeof(unsigned char) * 3;
    h_out_img = static_cast<unsigned char *>(malloc(mem_size));

    unsigned char *d_in_img, *d_out_img;
    CUDA_CHECK_THROW(cudaMallocAsync(&d_in_img, mem_size, stream))
    CUDA_CHECK_THROW(cudaMallocAsync(&d_out_img, mem_size, stream))

    CUDA_CHECK_THROW(
        cudaMemcpyAsync(d_in_img, h_in_img, mem_size, cudaMemcpyHostToDevice))

    size_t num_blocks = height * width / 1024;
    if ((height * width) % 1024 > 0)
        ++num_blocks;

    convolve_kernel<<<num_blocks,1024>>>(d_in_img, d_out_img, width, height);

    CUDA_CHECK_THROW(
        cudaMemcpyAsync(h_out_img, d_out_img, mem_size, cudaMemcpyDeviceToHost))

    CUDA_CHECK_THROW(cudaDeviceSynchronize())

    const auto t2 = duration_cast<milliseconds>(
        system_clock::now().time_since_epoch()).count();
    std::cout << "CUDA Runtime: " << t2 - t1 << std::endl;

    error = lodepng_encode24_file(
        CUDA_OUTPUT_FILENAME, h_out_img, width, height);
    if (error) {
        printf("error %u: %s\n", error, lodepng_error_text(error));
        exit(-1);
    }

    CUDA_CHECK_THROW(cudaFreeAsync(d_in_img, stream))
    CUDA_CHECK_THROW(cudaFreeAsync(d_out_img, stream))
    free(h_in_img);
    free(h_out_img);
}

void halideCUDAvecXY(const Func &blur, Var &x, Var &y, Var &c,
                     const Func &padded16, const Buffer<uint8_t> &input) {
    const Func blur_x, gaus;

    blur_x(x, y, c) = padded16(x - 2, y, c) + padded16(x - 1, y, c) +
                      padded16(x, y, c) + padded16(x + 1, y, c) +
                      padded16(x + 2, y, c);

    blur(x, y, c) = cast<uint8_t>(input(x, y, c) -
                                  (blur_x(x, y - 2, c) + blur_x(x, y - 1, c)
                                   + blur_x(x, y, c) + blur_x(x, y + 1, c) +
                                   blur_x(x, y + 2, c)) / 25);

}

void halideCUDARDom(const Func &blur, Var &x, Var &y, Var &c,
                    const Func &padded16, const Buffer<uint8_t> &input) {
    const Func kernel;
    RDom red(-2, 5, -2, 5);

    kernel(x, y) = 0;
    for (int xx = -2; xx <= 2; ++xx) {
        for (int yy = -2; yy <= 2; ++yy) {
            kernel(xx, yy) = 1;
        }
    }

    blur(x, y, c) = cast<uint8_t>(
        input(x, y, c) - sum(
            padded16(x + red.x, y + red.y, c) * kernel(red.x, red.y)) / 25);
}

void halideCUDA() {
    const Buffer<uint8_t> input = load_image(INPUT_FILENAME);
    assert(input.channels() == 3);

    Buffer<uint8_t> reference_output(input.width(), input.height(),
                                     input.channels());

    const auto t1 = duration_cast<milliseconds>(
        system_clock::now().time_since_epoch()).count();
    Var x, y, c;
    const Func padded, padded16;
    Func blur;

    padded(x, y, c) = input(clamp(x, 0, input.width() - 1),
                            clamp(y, 0, input.height() - 1), c);
    padded16(x, y, c) = cast<uint16_t>(padded(x, y, c));

    // halideCUDARDom(blur, x, y, c, padded16, input);
    halideCUDAvecXY(blur, x, y, c, padded16, input);

    const Var xo, yo, xi, yi;

    // blur.reorder(x, y, c).bound(c, 0, 3).unroll(c);

    // blur.split(y, yo, yi, 32).parallel(yo).bound(c, 0, 3).unroll(c);

    blur.gpu_tile(x, y, xo, yo, xi, yi, 16, 16).gpu_blocks(xo, yo).
         gpu_threads(xi, yi).bound(c, 0, 3).unroll(c);

    Target target = get_host_target();
    target = target.with_feature(Target::CUDA);
    assert(host_supports_target_device(target));

    blur.compile_jit(target);

    const auto t2 = duration_cast<milliseconds>(
        system_clock::now().time_since_epoch()).count();
    Buffer<uint8_t> output =
        blur.realize({input.width(), input.height(), input.channels()});

    CUDA_CHECK_THROW(cudaDeviceSynchronize())
    const auto t3 = duration_cast<milliseconds>(
        system_clock::now().time_since_epoch()).count();
    std::cout << "Halide Setup Time: " << t2 - t1 << std::endl;
    std::cout << "Halide Runtime: " << t3 - t2 << std::endl;
    std::cout << "Halide Total Time: " << t3 - t1 << std::endl;
    std::cout << "Halide Target " << target.to_string() << std::endl;

    save_image(output, HALIDE_OUTPUT_FILENAME);
}

int main() {
    // customCUDA();

    try {
        halideCUDA();
    } catch (Error e) {
        std::cout << e.what() << std::endl;
        exit(-1);
    }
}