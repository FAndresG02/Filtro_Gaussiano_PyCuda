import pycuda.driver as cuda  # Controlador CUDA
import pycuda.autoinit  # Inicializa automáticamente CUDA
from pycuda.compiler import SourceModule  # Para compilar código CUDA
import numpy as np  # Para operaciones con matrices
from PIL import Image  # Para manipular imágenes
import time  # Para medir el tiempo

# Genera un kernel gaussiano dado tamaño y sigma
def generar_kernel_gaussiano(tamaño, sigma):
    offset = tamaño // 2 
    kernel = np.zeros((tamaño, tamaño), dtype=np.float32)  
    for y in range(-offset, offset + 1):
        for x in range(-offset, offset + 1):
            # Calcula el valor gaussiano
            valor = (1 / (2 * np.pi * sigma ** 2)) * np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
            kernel[y + offset, x + offset] = valor
    kernel /= np.sum(kernel)  # Normaliza el kernel
    return kernel

# Código CUDA para aplicar el filtro gaussiano en GPU
gauss_kernel_code = """
__global__ void filtro_gauss(unsigned char* input, unsigned char* output, float* kernel, 
                             int ancho, int alto, int k_size) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;  // Coordenada X
    int y = blockIdx.y * blockDim.y + threadIdx.y;  // Coordenada Y

    int offset = k_size / 2;
    if (x >= ancho || y >= alto) return;  // Evita accesos fuera de rango

    float suma = 0.0f;
    for (int ky = -offset; ky <= offset; ky++) {
        for (int kx = -offset; kx <= offset; kx++) {
            int px = min(max(x + kx, 0), ancho - 1);  // Control de bordes X
            int py = min(max(y + ky, 0), alto - 1);   // Control de bordes Y
            float pixel = input[py * ancho + px];  // Valor del píxel
            float coeff = kernel[(ky + offset) * k_size + (kx + offset)];  // Valor del kernel
            suma += pixel * coeff;  // Convolución
        }
    }
    output[y * ancho + x] = min(max(int(suma), 0), 255);  // Asigna el resultado
}
"""

# Aplica el filtro gaussiano en GPU
def aplicar_filtro_gaussiano_gpu(gray_array, kernel):
    alto, ancho = gray_array.shape  # Dimensiones de la imagen
    kernel_size = kernel.shape[0]  # Tamaño del kernel

    input_flat = gray_array.astype(np.uint8).flatten()  # Aplana la imagen
    output_flat = np.empty_like(input_flat)  # Buffer para la salida
    kernel_flat = kernel.flatten().astype(np.float32)  # Aplana el kernel

    # Reserva memoria en GPU
    d_input = cuda.mem_alloc(input_flat.nbytes)  
    d_output = cuda.mem_alloc(output_flat.nbytes)
    d_kernel = cuda.mem_alloc(kernel_flat.nbytes)

    # Copia datos a GPU
    cuda.memcpy_htod(d_input, input_flat)  
    cuda.memcpy_htod(d_kernel, kernel_flat)

    #Compila el código CUDA y Obtiene función de GPU
    mod = SourceModule(gauss_kernel_code)  
    filtro_gauss = mod.get_function("filtro_gauss")   

    # Tamaño de bloque y de grilla
    block_size = (32, 32, 1)  
    grid_size = ((ancho + 31) // 32, (alto + 31) // 32)  

    # Ejecuta el kernel CUDA
    filtro_gauss(
        d_input, d_output, d_kernel,
        np.int32(ancho), np.int32(alto), np.int32(kernel_size),
        block=block_size, grid=grid_size
    )
    
    # Copia resultado a CPU y Reestructura la imagen
    cuda.memcpy_dtoh(output_flat, d_output)  
    return output_flat.reshape((alto, ancho))  

# Función principal
def main():
    inicio = time.time()  

    # Rutas de entrada y salida
    input_path = "C:/Users/andre/Documents/COMPUTACION_PARALELA/PYTHON/Tarea_Filtro_Python/img.jpg"
    output_path = "C:/Users/andre/Documents/COMPUTACION_PARALELA/PYTHON/Tarea_Filtro_PyCuda/img41.jpg"

    imagen = Image.open(input_path)  # Abre la imagen
    gray_array = np.array(imagen)  # Convierte a arreglo

    kernel_size = 41  # Tamaño del kernel
    sigma = 6.2  # Valor de sigma
    kernel = generar_kernel_gaussiano(kernel_size, sigma)  # Crea el kernel

    # Aplica el filtro en GPU
    resultado_array = aplicar_filtro_gaussiano_gpu(gray_array, kernel)  

    resultado_img = Image.fromarray(resultado_array.astype(np.uint8), mode='L')  
    resultado_img.save(output_path)  

    fin = time.time() 

    print("Filtro gaussiano aplicado con PyCUDA y un kernel de 41*41.")  
    print(f"Tiempo de ejecución: {int((fin - inicio) * 1000)} ms") 

if __name__ == "__main__":
    main()
