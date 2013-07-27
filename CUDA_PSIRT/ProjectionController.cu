#include "Projection.cu"

int** get_dummy_particle_per_projection_trajectory(int n_proj, int n_traj, int n_part);

Projection* new_parallel_projections(int n_projections, int n_trajectories, int** particles_per_trajectory)
{
	Projection* projections = (Projection*)malloc(sizeof(Projection)*n_projections);
	double angle_step = 180.0/(double)n_projections;
	double angle = 0;
	int i=0;

	for (i=0;i<n_projections;i++,angle+=angle_step)
	{
		projections[i] = *new_parallel_projection(angle, n_trajectories, particles_per_trajectory[i]);
	}

	return projections;
}

int** get_dummy_particle_per_projection_trajectory(int n_proj, int n_traj, int n_part)
{
	int **part_per_traj = (int**)malloc(sizeof(int)*n_proj);
	int i,j;

	for (i=0;i<n_proj;i++) //init
		part_per_traj[i] = (int*)malloc(sizeof(int)*n_traj);

	for (i=0;i<n_proj;i++)
		for (j=0;j<n_traj;j++)
			part_per_traj[i][j]=n_part;

	return part_per_traj;
}
