#include "Reconstruction.c"
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

void prep_psirt();

__global__ void psirt_kernel(PSIRT* dev_psirt);

int main()
{
	

    prep_psirt();

	
    
    
	printf("\r\nOK!");
	scanf("%d");

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    //cudaDeviceReset();
	
	
    return 0;
}

void prep_psirt() 
{
	// 1. INICIALIZAR (HOST)
	PSIRT* host_psirt = init_psirt();
	

	// 2. COPIAR (HOST->DEV)
	PSIRT* dev_psirt;
	cudaMalloc((void**)&dev_psirt, sizeof(PSIRT));
	cudaMemcpy(dev_psirt, host_psirt, sizeof(PSIRT), cudaMemcpyHostToDevice);

	// 3. EXECUTAR (DEV)
	psirt_kernel<<<1,1>>>(dev_psirt);

	// 4. COPIAR (DEV->HOST)
	cudaMemcpy(host_psirt, dev_psirt, sizeof(PSIRT), cudaMemcpyDeviceToHost);
	cudaFree(dev_psirt);

	// 5. RECONSTRUCAO (HOST)
	draw_reconstruction_bitmap(host_psirt);
	free(host_psirt);
}


__global__ void psirt_kernel(PSIRT* dev_psirt)
{
	while(1) run_psirt(dev_psirt);
}