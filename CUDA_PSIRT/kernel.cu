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

void prep_psirt();
void display();
void test_psirt();

__global__ void psirt_kernel(PSIRT* psirt, Projection** dev_proj);

PSIRT *debug_host_psirt;
PSIRT *dev_psirt;

Projection**	dev_projections;
Particle**		dev_particles;


__global__ void test(Trajectory* t, Particle* p, int* dev_params, PSIRT* dev_psirt)
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

void test_psirt(PSIRT* host_psirt)
{

	// 1: COPIAR PROJEÇÕES/TRAJETÓRIAS
	
	int n_proj = host_psirt->n_projections;
	int n_traj = host_psirt->n_trajectories;
	int n_ttl_traj = n_proj * n_traj;
	
	int ttl_traj_len = n_proj * n_traj;
	size_t ttl_traj_size = sizeof(Trajectory) * ttl_traj_len;

	Trajectory* traj;
	GPUerrchk(cudaMalloc((void**)&traj, ttl_traj_size));

	int i,j,k;
	
	for (i=0,k=0; i<n_proj; i++) 
	{
		for (j=0; j< n_traj; j++,k++)
		{
			Trajectory t = (host_psirt->trajectories[(i*host_psirt->n_projections)+j]);
			GPUerrchk(cudaMemcpy(&(traj[k]), &t, sizeof(Trajectory), cudaMemcpyHostToDevice));
		}
	}


	// 2. COPIAR PARTÍCULAS

	int n_part = host_psirt->n_particles;
	int ttl_part_size = sizeof(Particle) * n_part;

	Particle *part;
	GPUerrchk(cudaMalloc((void**)&part, ttl_part_size));

	for (i=0; i<n_part; i++)
	{
		Particle hp = host_psirt->particles[i];
		GPUerrchk(cudaMemcpy(&(part[i]), &hp, sizeof(Particle), cudaMemcpyHostToDevice));
	}

	// 3. DEMAIS PARAMETROS
	int params[] = {n_proj, n_traj, n_part};
	int *dev_params;
	GPUerrchk(cudaMalloc((void**)&dev_params, sizeof(int)*3));
	GPUerrchk(cudaMemcpy(dev_params,params,sizeof(int)*3,cudaMemcpyHostToDevice));

	
	// 3. REMONTAR PARAMETROS
	PSIRT* dev_psirt;
	Projection* dev_proj;
	GPUerrchk(cudaMalloc((void**)&dev_psirt, sizeof(PSIRT)));

	test<<<1,1>>>(traj, part, dev_params, dev_psirt);

	// 4. COPIAR DE VOLTA
	Particle *host_plist = host_psirt->particles;
	GPUerrchk(cudaMemcpy( host_plist, part, sizeof(Particle) * n_part, cudaMemcpyDeviceToHost));
}

int main(int argc, char* argv[])
{
	cudaError_t cudaStatus;

	PSIRT* host_psirt = init_psirt();	
	cudaError_t cuda_status = cudaSetDevice(0);
	if (cuda_status != cudaSuccess) printf("\r\n cuInit failed");

	//prep_psirt(); // CUDA

	test_psirt(host_psirt);

    
	//printf("\r\nOK!");
	//scanf("%d");

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }
	


	draw_projection_bitmap(host_psirt);
	draw_reconstruction_bitmap(host_psirt);
	
	free(host_psirt);

    return 0;
}