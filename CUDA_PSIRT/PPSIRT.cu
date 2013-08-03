#include "Reconstruction.c"
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define NPART 64
#define NTRAJ 21

// idéia: separar em NPART threads com PPSIRT-1 (atualizar particulas) + NTRAJ threads com PPSIRT-2 (atualizar trajetorias), asynch

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
	int optim_curr_iteration = 0;
	int optim_max_iterations = 100;

	int npart = dev_psirt->n_particles;
	int ttl_trajs = dev_psirt->n_trajectories * dev_psirt->n_projections;

	int done = 0;
	int lim = 0;

	double ttl_time_p1 = 0;
	double ttl_time_p2 = 0;

	__shared__ Particle sh_part[NPART];
	__shared__ float sh_dist_p_t[NPART][NTRAJ];

	sh_part[part_index] = p[part_index];
	

	while (!done)
	{
		if (part_index==0) lim++;
		
		// ---------------------------
		// *** ATUALIZAR POSICOES DAS PARTICULAS ***
		// ---------------------------
		int i=0,j=0;

		Vector2D resultant_force;
		set(&resultant_force,0.0,0.0);
		float rx=0.0, ry=0.0;
		for (i = 0; i < ttl_trajs; i++) 
		{
			sh_dist_p_t[part_index][i] = distance(&sh_part[part_index].location,&t[i]);	// calcular distancias
			resultant_(&(t[i]),&sh_part[part_index], &resultant_force, sh_dist_p_t[part_index][i]);
			rx+=resultant_force.x;
			ry+=resultant_force.y;
		}
	
		set(&resultant_force, -rx, -ry);
		update_particle(&sh_part[part_index], &resultant_force);
		

		
	

		// ---------------------------
		// *** CALCULO DE TRAJETORIAS SATISFEITAS ***
		// ---------------------------
		sh_part[part_index].current_trajectories = 0; 	// zera #traj de cada particula
		
		for (i=0;i<ttl_trajs; i++)  t[i].n_particulas_atual = 0;
		__syncthreads();

		for (i=0;i<ttl_trajs; i++) 
		{
			//if (distance(&sh_part[part_index].location,&t[i])<TRAJ_PART_THRESHOLD)
			if (sh_dist_p_t[part_index][i] < TRAJ_PART_THRESHOLD)
			{
				atomicAdd(&t[i].n_particulas_atual, 1);
				sh_part[part_index].current_trajectories++;
			}				
		}
		

		int stable = 0;
		for (i=0;i<ttl_trajs;i++)  if (t[i].n_particulas_atual>=t[i].n_particulas_estavel)	stable ++;
		
		__syncthreads();
		
		if (stable==ttl_trajs) // is stable					*************(trecho ok)
		{
			done = 1;
			// filtragem rápida
			if (sh_part[part_index].current_trajectories == 0 ) sh_part[part_index].status = DEAD;
		}

		
	}
	p[part_index] = sh_part[part_index];
	*iter = lim;
}

