#include "PSIRT.h"

void read_sinogram(PSIRT* psirt);

void read_sinogram(PSIRT* psirt)
{
	int n_parts;
	FILE * pFile;

	pFile = fopen ("sinograma.txt","r");

	fscanf (pFile, "%d", &psirt->n_projections);
	fscanf (pFile, "%d", &psirt->n_trajectories);
	psirt->projections = new_parallel_projections(psirt->n_projections,psirt->n_trajectories, get_dummy_particle_per_projection_trajectory(psirt->n_projections,psirt->n_trajectories,4));

	int i=0,j=0;
	for (i=0;i<psirt->n_projections;i++)
	{
		for (j=0;j<psirt->n_trajectories;j++)
		{
			fscanf (pFile, "%d", &n_parts);
			psirt->projections[i]->lista_trajetorias[j]->n_particulas_estavel = n_parts;
		}
	}

	fscanf (pFile, "%d", &psirt->n_particles);
	fclose (pFile);
}
