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

	printf("\r\n===========\r\nSTARTUP CUDA PSIRT\r\n===========\r\nPARAMS:");
	printf("\t(#PROJ)\t(#TRAJ)\t(NPART)\r\n\t%d\t%d\t%d\r\n\r\n",dev_psirt->n_projections, dev_psirt->n_trajectories, dev_psirt->n_particles);

	int done = 0;
	int lim = 0;

	while (!done)
	{
		++lim;
			// ---------------------------
		// *** ATUALIZAR POSICOES DAS PARTICULAS ***
		// ---------------------------
		int i=0,j=0;
		Vector2D resultant_force, resultant_vector;
		//for (i = 0; i < dev_psirt->n_particles; i++) 
		//{
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
		//}																// !!!!!!!!!!!!!!!!!!!!! paralelizar
		__syncthreads();
		// ---------------------------
		// *** CALCULO DE TRAJETORIAS SATISFEITAS ***
		// ---------------------------
		for (i=0;i<dev_psirt->n_particles;i++) dev_psirt->particles[i].current_trajectories = 0; 	// zera #traj de cada particula
		for (i=0;i<ttl_trajs; i++) 
		{
			t[i].n_particulas_atual = 0;
			for (j=0; j<npart; j++)																	// !!!!!!!!!!!!!!!!!!!!! paralelizar
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
		// ---------------------------
		// *** OTIMIZACAO E CONVERGENCIA ***
		// ---------------------------
		// pre-update otimizacao
		if (is_optimizing_dirty_particle) {
			// OTIMIZANDO
			if (optim_curr_iteration < optim_max_iterations) {
				optim_curr_iteration++;
			}
			// OTIMIZACAO FALHOU (EXCEDEU MAX ITERACOES)
			else {
				//			printf("\r\n[OPTIM]\tPARTICLE #%d LIVED (ITER #%d)",optim_curr_part,optim_curr_iteration);
				p[optim_curr_part].status = ALIVE; // NAO CONSEGUIU REMOVER
				optim_curr_part++;
				is_optimizing_dirty_particle = 0;
			}
		}
		__syncthreads();
		int stable = 0;
		for (i=0;i<ttl_trajs;i++) 
		{
			if (t[i].n_particulas_atual>=t[i].n_particulas_estavel)	stable ++;
		}

		if (lim>5000) {
			
		}

		if (stable==ttl_trajs) // is stable					*************(trecho ok)
		{
			// no optim
			done = 1;


			//if (optim_curr_part < npart) 
			//{
				// optimize
				// ---------------------------
				// *** PRIMEIRO PASSO ***
				// Deve-se ordenar as particulas de acordo
				// com a quantidade de trajetorias a que
				// cada uma atende (0 a 3), ordem crescente.
				// ---------------------------
				/*if (!optim_is_ranked)
				{
					Particle temp;
					for (i = 0; i < npart; i++)
					{
						for (j = npart-1; j > i; j--)
						{
							if ((p[j].current_trajectories < p[i].current_trajectories) & (i!=j))  
							{
								temp = p[i];
								p[i] = p[j];
								p[j] = temp;
							}
						}
					}
					optim_is_ranked = 1;
				}*/

				// ---------------------------
				// CASO ESPECIAL
				// partícula sem trajetoria, ELIMINAR SEM CHECAR
				// ---------------------------
				//__syncthreads();
			/*	if (p[part_index].current_trajectories == 0)
				{
					is_optimizing_dirty_particle = 0;
					p[part_index].status = DEAD;
					optim_curr_part++;
				}

				// ---------------------------
				// CASO NORMAL
				// partícula atende a mais de 0 trajetórias, CHECAR ANTES
				// ---------------------------
				else
				{
					is_optimizing_dirty_particle = 1;

					// COMECAR A CHECAR PARTICULA
					if (p[optim_curr_part].status == ALIVE)
					{
						optim_curr_iteration = 0;
						p[optim_curr_part].status = CHECKING;
					}
					// PARTICULA CHECADA & CONVERGIU -> REMOVER
					else if (p[optim_curr_part].status == CHECKING)
					{
						p[optim_curr_part].status = DEAD;
						optim_curr_part++;
					}
				}*/
			//}
			// (end optim)

			//else done = 1;	// DONE 
		}
	}
}








__global__ void run_cuda_psirt_singlethread(Trajectory* t, Particle* p, int* dev_params, PSIRT* dev_psirt)
{
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

	printf("\r\n===========\r\nSTARTUP CUDA PSIRT\r\n===========\r\nPARAMS:");
	printf("\t(#PROJ)\t(#TRAJ)\t(NPART)\r\n\t%d\t%d\t%d\r\n\r\n",dev_psirt->n_projections, dev_psirt->n_trajectories, dev_psirt->n_particles);

	int done = 0;
	int lim = 0;

	while (!done)
	{
		++lim;
			// ---------------------------
		// *** ATUALIZAR POSICOES DAS PARTICULAS ***
		// ---------------------------
		int i=0,j=0;
		Vector2D resultant_force, resultant_vector;
		for (i = 0; i < dev_psirt->n_particles; i++) 
		{
			if (p[i].status != DEAD) 
			{		
				set(&resultant_force,0.0,0.0);
				set(&resultant_vector,0.0,0.0);
				for (j = 0; j < ttl_trajs; j++) 
				{
					resultant(&(t[j]),&p[i], &resultant_vector);
					sum_void(&resultant_force, &resultant_vector, &resultant_force);
				}
				set(&resultant_force, -resultant_force.x, -resultant_force.y);
				update_particle(&p[i], &resultant_force);
			}
		}																// !!!!!!!!!!!!!!!!!!!!! paralelizar
		__syncthreads();
		// ---------------------------
		// *** CALCULO DE TRAJETORIAS SATISFEITAS ***
		// ---------------------------
		for (i=0;i<dev_psirt->n_particles;i++) dev_psirt->particles[i].current_trajectories = 0; 	// zera #traj de cada particula
		for (i=0;i<ttl_trajs; i++) 
		{
			t[i].n_particulas_atual = 0;
			for (j=0; j<npart; j++)																	// !!!!!!!!!!!!!!!!!!!!! paralelizar
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
		// ---------------------------
		// *** OTIMIZACAO E CONVERGENCIA ***
		// ---------------------------
		// pre-update otimizacao
		if (is_optimizing_dirty_particle) {
			// OTIMIZANDO
			if (optim_curr_iteration < optim_max_iterations) {
				optim_curr_iteration++;
			}
			// OTIMIZACAO FALHOU (EXCEDEU MAX ITERACOES)
			else {
				//			printf("\r\n[OPTIM]\tPARTICLE #%d LIVED (ITER #%d)",optim_curr_part,optim_curr_iteration);
				p[optim_curr_part].status = ALIVE; // NAO CONSEGUIU REMOVER
				optim_curr_part++;
				is_optimizing_dirty_particle = 0;
			}
		}
		__syncthreads();
		int stable = 0;
		for (i=0;i<ttl_trajs;i++) 
		{
			if (t[i].n_particulas_atual>=t[i].n_particulas_estavel)	stable ++;
		}

		if (lim>5000) {
			
		}

		if (stable==ttl_trajs) // is stable					*************(trecho ok)
		{
			// no optim
			done = 1;


			//if (optim_curr_part < npart) 
			//{
				// optimize
				// ---------------------------
				// *** PRIMEIRO PASSO ***
				// Deve-se ordenar as particulas de acordo
				// com a quantidade de trajetorias a que
				// cada uma atende (0 a 3), ordem crescente.
				// ---------------------------
				/*if (!optim_is_ranked)
				{
					Particle temp;
					for (i = 0; i < npart; i++)
					{
						for (j = npart-1; j > i; j--)
						{
							if ((p[j].current_trajectories < p[i].current_trajectories) & (i!=j))  
							{
								temp = p[i];
								p[i] = p[j];
								p[j] = temp;
							}
						}
					}
					optim_is_ranked = 1;
				}*/

				// ---------------------------
				// CASO ESPECIAL
				// partícula sem trajetoria, ELIMINAR SEM CHECAR
				// ---------------------------
				//__syncthreads();
			/*	if (p[part_index].current_trajectories == 0)
				{
					is_optimizing_dirty_particle = 0;
					p[part_index].status = DEAD;
					optim_curr_part++;
				}

				// ---------------------------
				// CASO NORMAL
				// partícula atende a mais de 0 trajetórias, CHECAR ANTES
				// ---------------------------
				else
				{
					is_optimizing_dirty_particle = 1;

					// COMECAR A CHECAR PARTICULA
					if (p[optim_curr_part].status == ALIVE)
					{
						optim_curr_iteration = 0;
						p[optim_curr_part].status = CHECKING;
					}
					// PARTICULA CHECADA & CONVERGIU -> REMOVER
					else if (p[optim_curr_part].status == CHECKING)
					{
						p[optim_curr_part].status = DEAD;
						optim_curr_part++;
					}
				}*/
			//}
			// (end optim)

			//else done = 1;	// DONE 
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


	// (parametros de paralelização)
	int n_elements = host_psirt->n_particles;
	int n_threads_per_block = 32;
	int n_blocks = n_elements/n_threads_per_block;

	cudaEvent_t start, start_paralel, stop_1, stop_paralel;
	cudaEventCreate(&start);
	cudaEventCreate(&start_paralel);
	cudaEventCreate(&stop_1);
	cudaEventCreate(&stop_paralel);
	
	cudaEventRecord(start);
	run_cuda_psirt_singlethread<<<1,1>>>(traj, part, dev_params, dev_psirt);
	cudaEventRecord(stop_1);
	
	

	// zerar
	cudaDeviceSynchronize();
	GPUerrchk(cudaMemcpy(traj, host_psirt->trajectories, sizeof(Trajectory) * n_ttl_traj, cudaMemcpyHostToDevice));
	GPUerrchk(cudaMemcpy(part, host_psirt->particles, sizeof(Particle) * n_part, cudaMemcpyHostToDevice));

	cudaEventRecord(start_paralel);
	run_cuda_psirt<<<n_blocks,n_threads_per_block>>>(traj, part, dev_params, dev_psirt);
	cudaEventSynchronize(stop_paralel);

	cudaDeviceSynchronize();


	float ms_1 = 0, ms_par = 0;
	cudaEventElapsedTime(&ms_1, start, stop_1);
	cudaEventElapsedTime(&ms_1, start_paralel, stop_paralel);
	cudaEventDestroy(start);
	cudaEventDestroy(start_paralel);
	cudaEventDestroy(stop_1);
	cudaEventDestroy(stop_paralel);

	printf ("\r\nFINALIZOU EXEC CUDA (1x1)\r\n TEMPO DE EXECUÇÃO FINAL: %f ms\r\n==============\r\n", ms_1);
	printf ("\r\nFINALIZOU EXEC CUDA (%dx%d)\r\n TEMPO DE EXECUÇÃO FINAL: %f ms\r\n==============\r\n", n_blocks, n_threads_per_block, ms_par);

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

	getchar();

    return 0;
}
