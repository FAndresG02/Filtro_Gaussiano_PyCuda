import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np
from PIL import Image
import time

def generar_kernel_gaussiano(tamaño, sigma):
    offset = tamaño // 2
    kernel = np.zeros((tamaño, tamaño), dtype=np.float32)
    for y in range(-offset, offset + 1):
        for x in range(-offset, offset + 1):
            valor = (1 / (2 * np.pi * sigma ** 2)) * np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
            kernel[y + offset, x + offset] = valor
    kernel /= np.sum(kernel)
    return kernel

gauss_kernel_code = """
__global__ void filtro_gauss(unsigned char* input, unsigned char* output, float* kernel, 
                             int ancho, int alto, int k_size) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int offset = k_size / 2;
    if (x >= ancho || y >= alto) return;

    float suma = 0.0f;
    for (int ky = -offset; ky <= offset; ky++) {
        for (int kx = -offset; kx <= offset; kx++) {
            int px = min(max(x + kx, 0), ancho - 1);
            int py = min(max(y + ky, 0), alto - 1);
            float pixel = input[py * ancho + px];
            float coeff = kernel[(ky + offset) * k_size + (kx + offset)];
            suma += pixel * coeff;
        }
    }
    output[y * ancho + x] = min(max(int(suma), 0), 255);
}
"""

def aplicar_filtro_gaussiano_gpu(gray_array, kernel):
    alto, ancho = gray_array.shape
    kernel_size = kernel.shape[0]

    input_flat = gray_array.astype(np.uint8).flatten()
    output_flat = np.empty_like(input_flat)
    kernel_flat = kernel.flatten().astype(np.float32)

    d_input = cuda.mem_alloc(input_flat.nbytes)
    d_output = cuda.mem_alloc(output_flat.nbytes)
    d_kernel = cuda.mem_alloc(kernel_flat.nbytes)

    cuda.memcpy_htod(d_input, input_flat)
    cuda.memcpy_htod(d_kernel, kernel_flat)

    mod = SourceModule(gauss_kernel_code)
    filtro_gauss = mod.get_function("filtro_gauss")

    block_size = (32, 32, 1)
    grid_size = ((ancho + 31) // 32, (alto + 31) // 32)

    filtro_gauss(
        d_input, d_output, d_kernel,
        np.int32(ancho), np.int32(alto), np.int32(kernel_size),
        block=block_size, grid=grid_size
    )

    cuda.memcpy_dtoh(output_flat, d_output)
    return output_flat.reshape((alto, ancho))

def main():
    inicio = time.time()

    input_path = "C:/Users/andre/Documents/COMPUTACION_PARALELA/PYTHON/Tarea_Filtro_Python/img.jpg"
    output_path = "C:/Users/andre/Documents/COMPUTACION_PARALELA/PYTHON/Tarea_Filtro_PyCuda/img41.jpg"

    imagen = Image.open(input_path)  
    gray_array = np.array(imagen)

    kernel_size = 41
    sigma = 6.2
    kernel = generar_kernel_gaussiano(kernel_size, sigma)

    resultado_array = aplicar_filtro_gaussiano_gpu(gray_array, kernel)

    resultado_img = Image.fromarray(resultado_array.astype(np.uint8), mode='L')
    resultado_img.save(output_path)

    fin = time.time()

    print("Filtro gaussiano aplicado con PyCUDA y un kernel de 41*41.")
    print(f"Tiempo de ejecución: {int((fin - inicio) * 1000)} ms")

if __name__ == "__main__":
    main()
