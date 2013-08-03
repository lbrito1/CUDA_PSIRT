#include "PPSIRT.cu"
#include <stdio.h>
#include <windows.h>
#include <time.h>

#define GPUerrchk(ans) { GPUassert((ans), __FILE__, __LINE__); }

//gambiarra
//#define DEBUG_PRINT

__global__ void run_cuda_psirt(Trajectory* t, Particle* p, int* dev_params, PSIRT* dev_psirt, int* iter);

void cuda_psirt(PSIRT* host_psirt);

__global__ void run_cuda_psirt(Trajectory* t, Particle* p, int* dev_params, PSIRT* dev_psirt, int* iter)
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

	while (!done)
	{
		if (part_index==0) lim++;
			// ---------------------------
		// *** ATUALIZAR POSICOES DAS PARTICULAS ***
		// ---------------------------
		int i=0,j=0;
		Vector2D resultant_force, resultant_vector;
		set(&resultant_force,0.0,0.0);
		set(&resultant_vector,0.0,0.0);
		for (j = 0; j < ttl_trajs; j++) 
		{
			resultant(&(t[j]),&p[part_index], &resultant_vector);
			sum_void(&resultant_force, &resultant_vector, &resultant_force);
		}
		set(&resultant_force, -resultant_force.x, -resultant_force.y);
		update_particle(&p[part_index], &resultant_force);
		

		
	

		// ---------------------------
		// *** CALCULO DE TRAJETORIAS SATISFEITAS ***
		// ---------------------------
		p[part_index].current_trajectories = 0; 	// zera #traj de cada particula
		
		for (i=0;i<ttl_trajs; i++)  t[i].n_particulas_atual = 0;
		__syncthreads();

		for (i=0;i<ttl_trajs; i++) 
		{
			if (distance(&p[part_index].location,&t[i])<TRAJ_PART_THRESHOLD)
			{
				atomicAdd(&t[i].n_particulas_atual, 1);
				p[part_index].current_trajectories++;
			}				
		}
		
		__syncthreads();
		int stable = 0;
		for (i=0;i<ttl_trajs;i++)  if (t[i].n_particulas_atual>=t[i].n_particulas_estavel)	stable ++;
		
		__syncthreads();
		
		if (stable==ttl_trajs) // is stable					*************(trecho ok)
		{
			done = 1;

		}

		
	}

	*iter = lim;
}








__global__ void run_cuda_psirt_singlethread(Trajectory* t, Particle* p, int* dev_params, PSIRT* dev_psirt, int* iter)
{
	*iter = 0;

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

	printf("\r\n===========\r\CUDA PSIRT (1x1)\r\n===========\r\nPARAMS:");
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
				if (distance(&p[j].location,&t[i])<TRAJ_PART_THRESHOLD)
				{
					t[i].n_particulas_atual++;
					p[j].current_trajectories++;
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
		for (i=0;i<ttl_trajs;i++) if (t[i].n_particulas_atual>=t[i].n_particulas_estavel)	stable ++;
		


		if (stable==ttl_trajs) // is stable					*************(trecho ok)
		{
			done = 1;
		}
	}

	*iter = lim;
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

	// PARAMETROS DE PROFILING
	int *dev_iter;
	GPUerrchk(cudaMalloc((void**)&dev_iter, sizeof(int)));


	// (parametros de paralelização)
	int n_elements = host_psirt->n_particles;
	int n_threads_per_block = 32;
	int n_blocks = n_elements/n_threads_per_block;

	cudaEvent_t start_1, start_paralel, stop_1, stop_paralel, start_cpu, stop_cpu;
	cudaEventCreate(&start_1);
	cudaEventCreate(&start_paralel);
	cudaEventCreate(&stop_1);
	cudaEventCreate(&stop_paralel);



	// CUDA 1x1 run
	cudaEventRecord(start_1,0);
	int iter_1x1 = 0;
	//run_cuda_psirt_singlethread<<<1,1>>>(traj, part, dev_params, dev_psirt, dev_iter);
	cudaDeviceSynchronize();
	cudaEventRecord(stop_1,0);
	cudaEventSynchronize(stop_1);
	GPUerrchk(cudaMemcpy(&iter_1x1, dev_iter, sizeof(int), cudaMemcpyDeviceToHost));


	

	// zerar
	GPUerrchk(cudaMemcpy(traj, host_psirt->trajectories, sizeof(Trajectory) * n_ttl_traj, cudaMemcpyHostToDevice));
	GPUerrchk(cudaMemcpy(part, host_psirt->particles, sizeof(Particle) * n_part, cudaMemcpyHostToDevice));
	

	// CUDA parallel run
	cudaEventRecord(start_paralel,0);
	int iter_par = 0;
	run_cuda_psirt<<<1, n_elements>>>(traj, part, dev_params, dev_psirt, dev_iter);
	cudaDeviceSynchronize();
	cudaEventRecord(stop_paralel,0);
	cudaEventSynchronize(stop_paralel);
	GPUerrchk(cudaMemcpy(&iter_par, dev_iter, sizeof(int), cudaMemcpyDeviceToHost));

	
	// CPU pure run
	clock_t start, end;
    float ms_cpu;
	int iter_cpu = 0;
	start = clock();
	while(!run_psirt_cpu_no_optim(host_psirt,iter_cpu++));		
	end = clock();
	ms_cpu = (float) (((double) (end - start)) / CLOCKS_PER_SEC);


	float ms_1 = 0, ms_par = 0;
	cudaEventElapsedTime(&ms_1, start_1, stop_1);
	cudaEventElapsedTime(&ms_par, start_paralel, stop_paralel);
	cudaEventDestroy(start_1);
	cudaEventDestroy(start_paralel);
	cudaEventDestroy(stop_1);
	cudaEventDestroy(stop_paralel);

	printf ("\r\nFINALIZOU EXEC CUDA (1x1)\r\n TEMPO DE EXECUCAO FINAL: %fms \t EM %d iters\r\n==============\r\n", ms_1, iter_1x1);
	printf ("\r\nFINALIZOU EXEC CUDA (%dx%d)\r\n TEMPO DE EXECUCAO FINAL: %fms \t EM %d iters\r\n==============\r\n", n_blocks, n_threads_per_block, ms_par, iter_par);
	printf("\r\nSpeedup intra-CUDA: %f\%\r\n",( (ms_1/ms_par)*100));
	printf ("\r\nFINALIZOU EXEC CPU\r\n TEMPO DE EXECUCAO FINAL: %fms \t EM %d iters\r\n==============\r\n", ms_cpu*1000, iter_cpu);

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
