#define PROJECTION_LENGTH 0.6

#include "Trajectory.cu"


// DATA STRUCTURES
typedef struct {
	Trajectory* lista_trajetorias;
} Projection;

// HOST FUNCTIONS
void init_proj(Projection* p, double angle, int ntraj, int* partptraj);

// DEVICE FUNCTIONS
__device__ int projection_stable(Projection *p, int n_traj);
__device__ int has_converged(Projection **p, int n_projections, int n_trajectories);

// IMPLEMENTATION
__device__ int projection_stable(Projection *p, int n_traj)
{
	int i=0, stable = 0;
	for (; i<n_traj; i++ ) stable += trajectory_stable(&p->lista_trajetorias[i]);
	if (stable==n_traj) return TRUE;
	else return FALSE;
}

__device__ int has_converged(Projection **p, int n_projections, int n_trajectories)
{
	int i=0, stable = 0;
	for (;i<n_projections;i++) stable += projection_stable(p[i], n_trajectories);
	if (stable==n_projections) return TRUE;
	else return FALSE;
}

Projection *new_parallel_projection(double angle, int ntraj, int* partptraj)
{
	Projection *p = (Projection *)malloc(sizeof(Projection));

	p->lista_trajetorias = (Trajectory*) malloc(ntraj*sizeof(Trajectory));

	Vector2D* director = new_vector(1.0f,0.0f);
	RotateClockWise(director,angle);

	Vector2D* ortogonal = new_vector(director->x, director->y);
	RotateClockWise(ortogonal,90.0f);
	normalize(ortogonal);

	Vector2D* begin, *end;

	begin = clone(ortogonal);
	end = clone(ortogonal);

	mult_constant_void(begin,(-PROJECTION_LENGTH/2));
	mult_constant_void(end, PROJECTION_LENGTH/2);

	int i=0;
	for(i=0;i<ntraj;i++)
	{
		Vector2D* center = new_vector(0.0,0.0);
		float coef = i/((float)(ntraj-1));
		lerp(begin,end,coef,center);
		Trajectory* traj = new_trajectory(*center,*director,partptraj[i]);

		p->lista_trajetorias[i] = *traj;
	}

	free(ortogonal);
	free(begin);
	free(end);

	return p;
}