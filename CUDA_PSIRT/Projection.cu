#define PROJECTION_LENGTH 0.6

#include "Trajectory.cu"

// DATA STRUCTURES
typedef struct {
	Trajectory* lista_trajetorias;
	int n_traj;
} Projection;

// HOST FUNCTIONS
void init_proj(Projection* p, double angle, int ntraj, int* partptraj);

// DEVICE FUNCTIONS
__device__ int projection_stable(Projection *p);
__device__ int has_converged(Projection **p, int n_projections);

// IMPLEMENTATION
__device__ int projection_stable(Projection *p)
{
	int i=0, stable = 0;
	for (; i<p->n_traj; i++ ) stable += trajectory_stable(&p->lista_trajetorias[i]);
	if (stable==p->n_traj) return TRUE;
	else return FALSE;
}

__device__ int has_converged(Projection **p, int n_projections)
{
	int i=0, stable = 0;
	for (;i<n_projections;i++) stable += projection_stable(p[i]);
	if (stable==n_projections) return TRUE;
	else return FALSE;
}

Projection *new_parallel_projection(double angle, int ntraj, int* partptraj)
{
	Projection *p = (Projection *)malloc(sizeof(Projection));

	p->n_traj = ntraj;
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