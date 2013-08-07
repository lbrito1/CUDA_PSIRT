#include "Reconstruction.c"
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define MAX_ITER 100

#define STATUS_STARTING 10
#define STATUS_RUNNING 0
#define STATUS_CONVERGED 1
#define STATUS_OPTIMIZING 2
#define STATUS_OPTIMIZED 3
#define STATUS_FINISHED 4

#define OPT_UNLOCKED -1


__global__ void dummy()
{
	int x = 0;
	x++;
}

__global__ void ppsirt(Trajectory* t, Particle* p, int* n_part, int* n_traj, int* iter, int* status, int* optim_lock, int* parts_optimized, int* optim_curr_iteration, int* stable)
{
	*iter = 0;
	// Indice da partícula a ser tratada nesta thread
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int tid_optim_status = STATUS_OPTIMIZING;
	__shared__ int lim;
	lim=0;
	atomicCAS(status, STATUS_STARTING, STATUS_RUNNING);

	if (*parts_optimized < *n_part)
	{
		__syncthreads();
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
			for (j = 0; j < *n_traj; j++) 
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
		p[tid].current_trajectories = 0; 	// zera #traj de cada particula
		for (i=0;i<*n_traj; i++) 
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
		
		
		
		// so tenta pegar lock se nao foi otimizada
		if (tid_optim_status != STATUS_OPTIMIZED & !( (p[tid].status == CHECKED | p[tid].status == DEAD) )  ) {
			atomicCAS(optim_lock, OPT_UNLOCKED, tid);	//tenta pegar o lock; quem conseguir comeca a otimizar 
		}

		// converged
		if (*stable==*n_traj) 
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
				if(++*optim_curr_iteration > MAX_ITER)	// nao convergiu sem a particula: manter
				{
					p[tid].status = CHECKED;
					atomicAdd(parts_optimized, 1);
					tid_optim_status = STATUS_OPTIMIZED;
					atomicCAS(status, STATUS_OPTIMIZING, STATUS_CONVERGED);
					atomicCAS(optim_lock, tid, -1); // terminou de otimizar: liberar lock
				}
			}
		}



		// checagem final para liberar lock
		if (p[tid].status == CHECKED | p[tid].status == DEAD) 
		{
			atomicCAS(optim_lock, tid, OPT_UNLOCKED);
		}
		
	}else {
		atomicExch(status, STATUS_FINISHED);
	}
	
	__syncthreads();
		
	

	*iter = lim;
}


__global__ void ppsirt_chkstable(Trajectory* t, int* stable)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	atomicExch(stable, 0);
	__syncthreads();
	if (t[tid].n_particulas_atual>=t[tid].n_particulas_estavel)	atomicAdd(stable, 1);
	__syncthreads();
}
