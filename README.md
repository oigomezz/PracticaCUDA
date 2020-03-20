# Practica CUDA

## Implementación del efecto borroso de una imagen

Con esta práctica se va a comparar el rendimiento de un algoritmo para la generación del efecto borroso de una imagen, bajo las siguientes condiciones de paralelización:

1. CPU - secuencial.
2. CPU - Hilos POSIX
3. CPU - Hilos OpenMP

El algoritmo y tipo de filtrado de imágenes usado es a través de una matriz de convolución. Es el tratamiento de una
matriz por otra que se llama kernel. El filtro de matriz de convolución usa la primera matriz que es la imagen que se va
a tratar. La imagen es una colección bidimensional de pı́xeles en coordenada rectangular. El kernel usado depende del efecto deseado.

## Comandos de ejecucion

nvcc blur-effect.cu -o blur-effect.out

./blur-effect.out
