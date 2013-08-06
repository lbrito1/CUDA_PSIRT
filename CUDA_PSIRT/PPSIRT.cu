#include "Reconstruction.c"
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define STATUS_STARTING -1
#define STATUS_RUNNING 0
#define STATUS_CONVERGED 1
#define STATUS_OPTIMIZING 2
#define STATUS_OPTIMIZED 3

#define NPART 64
#define NTRAJ 21


__global__ void ppsirt(Trajectory* t, Particle* p, int* dev_params, PSIRT* dev_psirt, int* iter)
{
	*iter = 0;

	// Indice da partícula a ser tratada nesta thread
	int part_index = blockIdx.x * blockDim.x + threadIdx.x;

	dev_psirt->particles = p;
	dev_psirt->trajectories = t;
	
	dev_psirt->n_projections = dev_params[0];
	dev_psirt->n_trajectories = dev_params[1];
	dev_psirt->n_particles = dev_params[2];

	int is_optimized = 0;
	int is_optimizing_dirty_particle = 0;
	int optim_is_ranked = 0;
	int optim_curr_part = 0;
	
	int optim_max_iterations = 100;

	Particle sh_p [NPART];
	sh_p[part_index] = p[part_index];
	__syncthreads();

	int npart = dev_psirt->n_particles;
	int ttl_trajs = dev_psirt->n_trajectories * dev_psirt->n_projections;

	int status = STATUS_STARTING;
	int lim = 0;

	double ttl_time_p1 = 0;
	double ttl_time_p2 = 0;

	while (status != STATUS_CONVERGED)
	{
		atomicAdd(&lim, 1);
			// ---------------------------
		// *** ATUALIZAR POSICOES DAS PARTICULAS ***
		// ---------------------------
		int i=0,j=0;
		Vector2D resultant_force, resultant_vector;
		if (sh_p[part_index].status != DEAD) 
		{		
			set(&resultant_force,0.0,0.0);
			set(&resultant_vector,0.0,0.0);
			for (j = 0; j < ttl_trajs; j++) 
			{
				resultant(&(t[j]),&sh_p[part_index], &resultant_vector);
				sum_void(&resultant_force, &resultant_vector, &resultant_force);
			}
			set(&resultant_force, -resultant_force.x, -resultant_force.y);
			update_particle(&sh_p[part_index], &resultant_force);
		}
	
		__syncthreads();
	

		// ---------------------------
		// *** CALCULO DE TRAJETORIAS SATISFEITAS ***
		// ---------------------------
		dev_psirt->particles[part_index].current_trajectories = 0; 	// zera #traj de cada particula
		for (i=0;i<ttl_trajs; i++) 
		{
			t[i].n_particulas_atual = 0;
			float distance_point_line = distance(&sh_p[part_index].location,&t[i]);
			if (distance_point_line<TRAJ_PART_THRESHOLD)
			{
				atomicAdd(&(t[i].n_particulas_atual), 1);
				sh_p[part_index].current_trajectories++;
			}
		}
		
		__syncthreads();
		int stable = 0;
		for (i=0;i<ttl_trajs;i++)  if (t[i].n_particulas_atual>=t[i].n_particulas_estavel)	stable ++;
		

		
		if (stable==ttl_trajs) // is stable					*************(trecho ok)
		{
			status = STATUS_CONVERGED;

		}

		//otimizar
	//	if (status == STATUS_CONVERGED) status = STATUS_OPTIMIZING;
		if (status == STATUS_OPTIMIZING)
		{
		//	int max
		}

		

		
	}

	p[part_index] = sh_p[part_index];
	__syncthreads();

	*iter = lim;
}


