#include "Reconstruction.c"
#include "device_launch_parameters.h"
#include "cuda.h"
#include "cuda_runtime.h"

__global__ void run_cuda_psirt_2(Trajectory* t, Particle* p, int* dev_params, int* iter)
{
	*iter = 0;

	// Indice da partícula a ser tratada nesta thread
	int part_index = blockIdx.x * blockDim.x + threadIdx.x;

	int n_projections = dev_params[0];
	int n_trajectories = dev_params[1];
	int n_particles = dev_params[2];
	int ttl_trajs = n_trajectories * n_projections;
	
	if (part_index==0) {
		printf("\r\n===========\r\nSTARTUP CUDA PSIRT (PARALLEL)\r\n===========\r\nPARAMS:");
		printf("\t(#PROJ)\t(#TRAJ)\t(NPART)\r\n\t%d\t%d\t%d\r\n\r\n",n_projections, n_trajectories, n_particles);
	}

	int done = 0;
	int lim = 0;

	while (!done)
	{
		if (part_index==0) lim++;
			// ---------------------------
		// *** ATUALIZAR POSICOES DAS PARTICULAS ***
		// ---------------------------
		int i=0,j=0;
		Vector2D resultant_force, resultant_vector;
		if (p[part_index].status != DEAD) 
		{		
			set(&resultant_force,0.0,0.0);
			set(&resultant_vector,0.0,0.0);
			for (j = 0; j < ttl_trajs; j++) 
			{
				resultant(&(t[j]),&p[part_index], &resultant_vector);
				sum_void(&resultant_force, &resultant_vector, &resultant_force);
			}
			set(&resultant_force, -resultant_force.x, -resultant_force.y);
			update_particle(&p[part_index], &resultant_force);
		}

		p[part_index].current_trajectories = 0; 	// zera #traj de cada particula

		__syncthreads();
		// ---------------------------
		// *** CALCULO DE TRAJETORIAS SATISFEITAS ***
		// ---------------------------
		
		for (i=0;i<ttl_trajs; i++) 
		{
			t[i].n_particulas_atual = 0;
			for (j=0; j<n_particles; j++)																	// !!!!!!!!!!!!!!!!!!!!! paralelizar
			{
				if (p[j].status == ALIVE)
				{
					float distance_point_line = distance(&p[j].location,&t[i]);
					if (distance_point_line<TRAJ_PART_THRESHOLD)
					{
						t[i].n_particulas_atual++;
						p[j].current_trajectories++;
					}
				}
				
			}
		}
		
		__syncthreads();
		int stable = 0;
		for (i=0;i<ttl_trajs;i++)  if (t[i].n_particulas_atual>=t[i].n_particulas_estavel)	stable ++;
		

		
		if (stable==ttl_trajs) // is stable					*************(trecho ok)
		{
			done = 1;

		}
	}

	*iter = lim;
}


