#include "Reconstruction.c"
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <windows.h>
#include <time.h>

#define GPUerrchk(ans) { GPUassert((ans), __FILE__, __LINE__); }

//gambiarra
//#define DEBUG_PRINT

__global__ void run_cuda_psirt(Trajectory* t, Particle* p, int* dev_params, PSIRT* dev_psirt);

void cuda_psirt(PSIRT* host_psirt);

__global__ void run_cuda_psirt(Trajectory* t, Particle* p, int* dev_params, PSIRT* dev_psirt)
{
	//printf("\r\n===========\r\nSTARTUP CUDA PSIRT\r\n===========\r\nPARAMS:");
	//printf("\t(#PROJ)\t(#TRAJ)\t(NPART)\r\n\t%d\t%d\t%d",dev_params[0], dev_params[1], dev_params[2]);

	dev_psirt->particles = p;
	dev_psirt->trajectories = t;
	
	dev_psirt->n_projections = dev_params[0];
	dev_psirt->n_trajectories = dev_params[1];
	dev_psirt->n_particles = dev_params[2];

	dev_psirt->is_optimized = 0;
	dev_psirt->is_optimizing_dirty_particle = 0;
	dev_psirt->optim_is_ranked = 0;
	dev_psirt->optim_curr_part = 0;
	dev_psirt->optim_curr_iteration = 0;
	dev_psirt->optim_max_iterations = 100;

	int ttl_trajs = dev_psirt->n_trajectories * dev_psirt->n_projections;

	int is_optimizing_dirty_particle = 0;


	printf("\r\n===========\r\nSTARTUP CUDA PSIRT\r\n===========\r\nPARAMS:");
	printf("\t(#PROJ)\t(#TRAJ)\t(NPART)\r\n\t%d\t%d\t%d\r\n\r\n",dev_psirt->n_projections, dev_psirt->n_trajectories, dev_psirt->n_particles);



	int done = 0;
	int lim = 0;


	while (!done&(++lim<1000))
	{
			// ---------------------------
		// *** ATUALIZAR POSICOES DAS PARTICULAS ***
		// ---------------------------
		update_particles(dev_psirt);

		// ---------------------------
		// *** CALCULO DE TRAJETORIAS SATISFEITAS ***
		// ---------------------------
		int i=0,j=0;
		for (i=0;i<dev_psirt->n_particles;i++) dev_psirt->particles[i].current_trajectories = 0; 	// zera #traj de cada particula
		for (i=0;i<ttl_trajs; i++) 											// calcula #traj de cada particula
			update_trajectory(dev_psirt->trajectories[i], dev_psirt->particles, dev_psirt->n_particles);

		// ---------------------------
		// *** OTIMIZACAO E CONVERGENCIA ***
		// ---------------------------
		optimization_check(dev_psirt);													// otimizacao (continuar)

		if (has_converged(dev_psirt->trajectories,dev_psirt->n_projections*dev_psirt->n_trajectories))							// convergiu
		{
			if (dev_psirt->optim_curr_part < dev_psirt->n_particles) optimize(dev_psirt);						// otimizacao (comecar)
			else done = 1;	// DONE 
		}
	}
}

void cuda_psirt(PSIRT* host_psirt)
{
	// 1: COPIAR PROJEÇÕES/TRAJETÓRIAS/PARTÍCULAS
	
	int n_proj = host_psirt->n_projections;
	int n_traj = host_psirt->n_trajectories;
	int n_ttl_traj = n_proj * n_traj;
	int n_part = host_psirt->n_particles;
	
	Trajectory* traj;
	GPUerrchk(cudaMalloc((void**)&traj, sizeof(Trajectory) * n_ttl_traj));
	GPUerrchk(cudaMemcpy(traj, host_psirt->trajectories, sizeof(Trajectory) * n_ttl_traj, cudaMemcpyHostToDevice));

	Particle *part;
	GPUerrchk(cudaMalloc((void**)&part, n_part * sizeof(Particle)));
	GPUerrchk(cudaMemcpy(part, host_psirt->particles, sizeof(Particle) * n_part, cudaMemcpyHostToDevice));

	// 2. PARAMETROS AUXILIARES
	int params[] = {n_proj, n_traj, n_part};
	int *dev_params;
	GPUerrchk(cudaMalloc((void**)&dev_params, sizeof(int)*3));
	GPUerrchk(cudaMemcpy(dev_params,params,sizeof(int)*3,cudaMemcpyHostToDevice));
	
	// 3. EXECUTAR
	PSIRT* dev_psirt;
	GPUerrchk(cudaMalloc((void**)&dev_psirt, sizeof(PSIRT)));

	run_cuda_psirt<<<1,1>>>(traj, part, dev_params, dev_psirt);

	// 4. COPIAR DE VOLTA
	Particle *host_plist = host_psirt->particles;
	GPUerrchk(cudaMemcpy( host_plist, part, sizeof(Particle) * n_part, cudaMemcpyDeviceToHost));
}

int main(int argc, char* argv[])
{
	// Inicializar CUDA
	GPUerrchk(cudaSetDevice(0));

	// Preparar parâmetros no host
	PSIRT* host_psirt = init_psirt();	

	// Passar parâmetros para device, executar & copiar de volta para host
	cuda_psirt(host_psirt);
	
	// Gerar bitmaps
	draw_projection_bitmap(host_psirt);
	draw_reconstruction_bitmap(host_psirt);
	
	// Limpeza & finalização
	free(host_psirt);
	GPUerrchk(cudaDeviceReset());
    return 0;
}