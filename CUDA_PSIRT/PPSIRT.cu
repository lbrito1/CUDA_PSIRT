#include "Reconstruction.c"
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define STATUS_STARTING 10
#define STATUS_RUNNING 0
#define STATUS_CONVERGED 1
#define STATUS_OPTIMIZING 2
#define STATUS_OPTIMIZED 3

#define OPT_UNLOCKED -1


__global__ void dummy(int* x)
{
	atomicAdd(x,1);
	atomicCAS(x,1,10);
	atomicCAS(x,10,11);
	atomicCAS(x,11,12);
}

__global__ void ppsirt(Trajectory* t, Particle* p, int* dev_params, PSIRT* dev_psirt, int* iter, int* status, int* optim_lock, int* parts_optimized, int* stable)
{
	*iter = 0;

	// Indice da partícula a ser tratada nesta thread
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

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

	double ttl_time_p1 = 0;
	double ttl_time_p2 = 0;

	int tid_optim_status = STATUS_OPTIMIZING;

	__shared__ int lim;
	lim=0;

	atomicCAS(status, STATUS_STARTING, STATUS_RUNNING);


	while (*parts_optimized < dev_psirt->n_particles)
	{
		atomicAdd(&lim, 1);
			// ---------------------------
		// *** ATUALIZAR POSICOES DAS PARTICULAS ***
		// ---------------------------
		int i=0,j=0;
		Vector2D resultant_force, resultant_vector;
		if (p[tid].status != DEAD) 
		{		
			set(&resultant_force,0.0,0.0);
			set(&resultant_vector,0.0,0.0);
			for (j = 0; j < ttl_trajs; j++) 
			{
				resultant(&(t[j]),&p[tid], &resultant_vector);
				sum_void(&resultant_force, &resultant_vector, &resultant_force);
			}
			set(&resultant_force, -resultant_force.x, -resultant_force.y);
			update_particle(&p[tid], &resultant_force);
		}
		
		// ---------------------------
		// *** CALCULO DE TRAJETORIAS SATISFEITAS ***
		// ---------------------------
		dev_psirt->particles[tid].current_trajectories = 0; 	// zera #traj de cada particula
		for (i=0;i<ttl_trajs; i++) 
		{
			if (p[tid].status == ALIVE | p[tid].status == CHECKED) 
			{
				t[i].n_particulas_atual = 0;
				float distance_point_line = distance(&p[tid].location,&t[i]);
				if (distance_point_line<TRAJ_PART_THRESHOLD)
				{
					atomicAdd(&(t[i].n_particulas_atual), 1);
					p[tid].current_trajectories++;
				}
			}
		}
		
		atomicExch(stable, 0);
		__syncthreads();
		if (tid<ttl_trajs) if (t[tid].n_particulas_atual>=t[tid].n_particulas_estavel)	atomicAdd(stable, 1);
		
		// so tenta pegar lock se nao foi otimizada
		if (tid_optim_status != STATUS_OPTIMIZED) {
			atomicCAS(optim_lock, OPT_UNLOCKED, tid);	//tenta pegar o lock; quem conseguir comeca a otimizar 
		}

	
		__syncthreads();
		// converged
		if (*stable==ttl_trajs) 
		{
			atomicCAS(status, STATUS_RUNNING, STATUS_CONVERGED);	// comecou a otimizar agora

			if (*status == STATUS_OPTIMIZING && *optim_lock==tid) // convergiu sem a particula: remover
			{
				p[tid].status = DEAD;
				atomicAdd(parts_optimized, 1);
				tid_optim_status = STATUS_OPTIMIZED;
				atomicCAS(status, STATUS_OPTIMIZING, STATUS_CONVERGED);
				atomicCAS(optim_lock, tid, OPT_UNLOCKED); // terminou de otimizar: liberar lock
			}
			else if (*status == STATUS_CONVERGED && *optim_lock==tid && p[tid].status == ALIVE)	// comecar a otimizar
			{
					p[tid].status = CHECKING;
					atomicCAS(status, STATUS_CONVERGED, STATUS_OPTIMIZING);
					
			}
		}
		// did not converge
		else 	
		{
			if (*status == STATUS_OPTIMIZING && *optim_lock==tid) 
			{
				if(++optim_curr_iteration > optim_max_iterations)	// nao convergiu sem a particula: manter
				{
					p[tid].status = CHECKED;
					atomicAdd(parts_optimized, 1);
					tid_optim_status = STATUS_OPTIMIZED;
					atomicCAS(status, STATUS_OPTIMIZING, STATUS_CONVERGED);
					atomicCAS(optim_lock, tid, -1); // terminou de otimizar: liberar lock
				}
			}
		}
		__syncthreads();
		
	}
	__syncthreads();
	*iter = lim;
}


