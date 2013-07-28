#include "ProjectionController.cu"
#include <cstdio>
#include <stdlib.h>
#define DEBUG_PRINT
typedef struct {
	Trajectory* trajectories;
	Particle* particles;

	int n_projections;
	int n_trajectories;
	int n_particles;

	int is_optimized;
	int is_optimizing_dirty_particle;
	int optim_is_ranked;

	int optim_curr_part;
	int optim_curr_iteration;
	int optim_max_iterations;
} PSIRT;

// HOST FUNCITONS
__host__ PSIRT* init_psirt();
void init_particles(PSIRT* psirt);
void read_sinogram(PSIRT* psirt);


// DEVICE FUNCTIONS
__device__ int update_particles(PSIRT* psirt);
__device__ void optimize(PSIRT* psirt);
__device__ int run_psirt(PSIRT* psirt);
__device__ void optimization_check(PSIRT* psirt);

__device__ PSIRT* get_psirt_dev(
	Projection** projections,
	Particle** particles,
	int n_projections,
	int n_trajectories,
	int n_particles,
	int is_optimized,
	int is_optimizing_dirty_particle,
	int optim_is_ranked,
	int optim_curr_part,
	int optim_curr_iteration,
	int optim_max_iterations
	);

__host__ __device__ void set_psirt(PSIRT* psirt);
/*

__device__ PSIRT* get_psirt_dev(
	Projection** projections,
	Particle** particles,
	int n_projections,
	int n_trajectories,
	int n_particles,
	int is_optimized,
	int is_optimizing_dirty_particle,
	int optim_is_ranked,
	int optim_curr_part,
	int optim_curr_iteration,
	int optim_max_iterations
	)
{
	PSIRT *psirt;
	cudaMalloc((void**)&psirt,sizeof(PSIRT));

	psirt->projections = projections;
	psirt->particles	= particles;
	psirt->n_projections = n_projections;
	psirt->n_trajectories = n_trajectories;
	psirt->n_particles = n_particles;
	psirt->is_optimized = is_optimized;
	psirt->is_optimizing_dirty_particle = is_optimizing_dirty_particle;
	psirt->optim_curr_part = optim_curr_part;
	psirt->optim_curr_iteration = optim_curr_iteration;
	psirt->optim_max_iterations = optim_max_iterations;

	return psirt;
}
*/

__host__ PSIRT* init_psirt()
{
	//gettimeofday(&tv, NULL);
	//curtime=tv.tv_usec;

	PSIRT *psirt;
	psirt = (PSIRT*)malloc(sizeof(PSIRT));
	psirt->is_optimized = 0;
	psirt->is_optimizing_dirty_particle = 0;
	psirt->optim_is_ranked = 0;
	psirt->optim_curr_part = 0;
	psirt->optim_curr_iteration = 0;
	psirt->optim_max_iterations = 100;
	//printf("\r\n\r\ns>>>%d ",psirt->optim_max_iterations);	
	read_sinogram(psirt);
	init_particles(psirt);

	//printf("\r\n[DONE] \t INIT \t %d us",(timestep=(tv.tv_usec-curtime)));

	return psirt;
}

__device__ int run_psirt(PSIRT* psirt)
{
	// ---------------------------
	// *** ATUALIZAR POSICOES DAS PARTICULAS ***
	// ---------------------------
	update_particles(psirt);

	// ---------------------------
	// *** CALCULO DE TRAJETORIAS SATISFEITAS ***
	// ---------------------------
	int i=0,j=0;
	for (i=0;i<psirt->n_particles;i++) psirt->particles[i].current_trajectories = 0; 	// zera #traj de cada particula
	for (i=0;i<psirt->n_projections;i++) 											// calcula #traj de cada particula
		for (j=0;j<psirt->n_trajectories;j++)
			update_trajectory(&(psirt->trajectories[(i*psirt->n_projections)+j]), &psirt->particles, psirt->n_particles);

	// ---------------------------
	// *** OTIMIZACAO E CONVERGENCIA ***
	// ---------------------------
	optimization_check(psirt);													// otimizacao (continuar)
	if (has_converged(psirt->trajectories,psirt->n_projections*psirt->n_trajectories))							// convergiu
	{
		if (psirt->optim_curr_part < psirt->n_particles) optimize(psirt);						// otimizacao (comecar)
		else return 0;	// DONE
	}
	return 1;	// NOT DONE
}

__device__ void optimize(PSIRT* psirt)
{
	int i = 0, j = 0;

	// ---------------------------
	// *** PRIMEIRO PASSO ***
	// Deve-se ordenar as particulas de acordo
	// com a quantidade de trajetorias a que
	// cada uma atende (0 a 3), ordem crescente.
	// ---------------------------
	if (!psirt->optim_is_ranked)
	{
		Particle temp;
		for (i = 0; i < psirt->n_particles; i++)
		{
			for (j = 0; j < psirt->n_particles; j++)
			{
				if (psirt->particles[j].current_trajectories
						> psirt->particles[i].current_trajectories) {
					memcpy(&temp, &psirt->particles[j], sizeof(Particle));
					memcpy(&psirt->particles[j], &psirt->particles[i], sizeof(Particle));
					memcpy(&psirt->particles[i], &temp, sizeof(Particle));
				}
			}
		}
		psirt->optim_is_ranked = 1;
	}

	// ---------------------------
	// CASO ESPECIAL
	// partícula sem trajetoria, ELIMINAR SEM CHECAR
	// ---------------------------
	if (psirt->particles[psirt->optim_curr_part].current_trajectories == 0)
	{
		//printf("\r\n[OPTIM]\tPARTICLE #%d DIED (0 TRAJ)", optim_curr_part);
		psirt->is_optimizing_dirty_particle = 0;
		psirt->particles[psirt->optim_curr_part].status = DEAD;
		psirt->optim_curr_part++;
	}

	// ---------------------------
	// CASO NORMAL
	// partícula atende a mais de 0 trajetórias, CHECAR ANTES
	// ---------------------------
	else
	{
		psirt->is_optimizing_dirty_particle = 1;

		// COMECAR A CHECAR PARTICULA
		if (psirt->particles[psirt->optim_curr_part].status == ALIVE)
		{
			psirt->optim_curr_iteration = 0;
			psirt->particles[psirt->optim_curr_part].status = CHECKING;
		}
		// PARTICULA CHECADA & CONVERGIU -> REMOVER
		else if (psirt->particles[psirt->optim_curr_part].status == CHECKING)
		{
			//printf("\r\n[OPTIM]\tPARTICLE #%d DIED (ITER #%d)", optim_curr_part,optim_curr_iteration);
			psirt->particles[psirt->optim_curr_part].status = DEAD;
			psirt->optim_curr_part++;
		}
	}
}


__device__ void optimization_check(PSIRT* psirt)
{
	// pre-update otimizacao
	if (psirt->is_optimizing_dirty_particle) {
		// OTIMIZANDO
		if (psirt->optim_curr_iteration < psirt->optim_max_iterations) {
			psirt->optim_curr_iteration++;
		}
		// OTIMIZACAO FALHOU (EXCEDEU MAX ITERACOES)
		else {
			//			printf("\r\n[OPTIM]\tPARTICLE #%d LIVED (ITER #%d)",optim_curr_part,optim_curr_iteration);
			psirt->particles[psirt->optim_curr_part].status = ALIVE; // NAO CONSEGUIU REMOVER
			psirt->optim_curr_part++;
			psirt->is_optimizing_dirty_particle = 0;
		}
	}
}

__device__ int update_particles(PSIRT* psirt)
{
	int i=0,j=0,k=0;
	Vector2D resultant_force;
	for (i = 0; i < psirt->n_particles; i++) {
		printf("\r\ni=%d",i);
		if (psirt->particles[i].status != DEAD) {
				
			set(&resultant_force,0.0,0.0);
			// calcular força resultante primeiro
			Vector2D resultant_vector;
			set(&resultant_vector,0.0,0.0);
			for (j = 0; j < psirt->n_projections; j++) {
				for (k = 0; k < psirt->n_trajectories; k++) {
					resultant(&(psirt->trajectories[(i*psirt->n_projections)+j]),&psirt->particles[i], &resultant_vector);
					sum_void(&resultant_force, &resultant_vector, &resultant_force);
				}

			}
#ifdef DEBUG_PRINT
			printf("\r\nPART [%d] \t pos (%f, %f)", i, psirt->particles[i].location.x, psirt->particles[i].location.y );
#endif
			set(&resultant_force, -resultant_force.x, -resultant_force.y);

			update_particle(&psirt->particles[i], &resultant_force);
		}
	}
	return i;
}

// ---------------------------
// *** INICIALIZAR PARTICULAS ***
// ---------------------------
void init_particles(PSIRT* psirt) {
	//srand(time(NULL ));
	int i = 0;
	int lim = RAND_MAX / 2;
	psirt->particles = (Particle*)malloc(sizeof(Particle) * psirt->n_particles);
	for (i = 0; i < psirt->n_particles; i++) {
		psirt->particles[i] = *new_particle();
		psirt->particles[i].location.x = rand() / (double) RAND_MAX;
		psirt->particles[i].location.y = rand() / (double) RAND_MAX;
		if (rand() > lim)
			psirt->particles[i].location.x = -psirt->particles[i].location.x;
		if (rand() > lim)
			psirt->particles[i].location.y = -psirt->particles[i].location.y;
	}

}


void read_sinogram(PSIRT* psirt)
{
	int n_parts;
	int ttl_traj = psirt->n_projections * psirt->n_trajectories;
	FILE * pFile;

	Projection *ptemp;
	
	pFile = fopen ("sinograma.txt","r");
	if (pFile != NULL) 
	{
		
		fscanf (pFile, "%d", &psirt->n_projections);
		fscanf (pFile, "%d", &psirt->n_trajectories);
		ptemp = new_parallel_projections(psirt->n_projections,psirt->n_trajectories, get_dummy_particle_per_projection_trajectory(psirt->n_projections,psirt->n_trajectories,4));

		int i=0,j=0, k=0;
		for (i=0;i<psirt->n_projections;i++)
		{
			for (j=0;j<psirt->n_trajectories;j++)
			{
				fscanf (pFile, "%d", &n_parts);
				ptemp[i].lista_trajetorias[j].n_particulas_estavel = n_parts;
			}
		}

		fscanf (pFile, "%d", &psirt->n_particles);
		fclose (pFile);

		int ttl_traj = (psirt->n_projections * psirt->n_trajectories);

		psirt->trajectories = (Trajectory*) malloc(sizeof(Trajectory) * ttl_traj);
		printf("\r\n\r\nREADING SINOGRAM\r\n");

		
		for (k=0,i=0;i<psirt->n_projections;i++)
		{
			for (j=0;j<psirt->n_trajectories;j++, k++)
			{
				memcpy(&(psirt->trajectories[k]), &(ptemp[i].lista_trajetorias[j]), sizeof(Trajectory));
				Trajectory t = psirt->trajectories[k];
				printf("\r\nTRAJ #%d\t%f,%f\t%f,%f",k,t.source.x,t.source.y,t.direction.x,t.direction.y);
			}
		}
	}
	else
	{
		printf("\r\nERROR: FILE NOT FOUND");
	}
}


