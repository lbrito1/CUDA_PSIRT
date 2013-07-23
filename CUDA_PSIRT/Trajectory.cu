/*
 * Trajectory.h
 *
 *  Created on: Apr 1, 2013
 *      Author: Leo
 */

#ifndef FALSE
#define FALSE 0
#endif

#ifndef TRUE
#define TRUE 1
#endif

#ifndef TRAJ_PART_THRESHOLD
#define TRAJ_PART_THRESHOLD 0.015
#endif

#ifndef TRAJ_FORCE_LINEAR
#define TRAJ_FORCE_LINEAR 3
#endif

#ifndef TRAJ_FORCE_POLY
#define TRAJ_FORCE_POLY 2
#endif

#define max(a,b) \
		({ __typeof__ (a) _a = (a); \
		__typeof__ (b) _b = (b); \
		_a > _b ? _a : _b; })

/*
 * Trajetoria.h
 *
 *  Created on: Mar 25, 2013
 *      Author: Leo
 */

#include "Particle.cu"

// DATA STRUCTURES

typedef struct {
	Vector2D source;
	Vector2D direction;
	int n_particulas_estavel;
	int n_particulas_atual;
} Trajectory;

// HOST FUNCTIONS
Trajectory* new_trajectory(Vector2D s, Vector2D d, int part_est);


// DEVICE FUNCTIONS
__device__ int trajectory_stable(Trajectory *t);
__device__ Vector2D* trajectory_intersection(Trajectory *t1, Trajectory *t2);
__device__ float distance(Vector2D* p, Trajectory* t);
__device__ Vector2D* projection(Vector2D* p, Trajectory* t);
__device__ int current_status(Trajectory* t);
__device__ float trajectory_force(Trajectory* t);
__device__ inline void resultant(Trajectory *t, Particle* p, Vector2D *resultant);
__device__ void update_trajectory(Trajectory *t, Particle **p, int nparticle);
__device__ inline void directionFrom(Vector2D *point, Trajectory *t, Vector2D *direction);




// IMPLEMENTATION
Trajectory* new_trajectory(Vector2D s, Vector2D d, int part_est) {
	Trajectory *t = (Trajectory*)malloc(sizeof(Trajectory));
	t->source=s;
	t->direction=d;
	t->n_particulas_estavel = part_est;
	t->n_particulas_atual = 0;
	return t;
}

__device__ float distance(Vector2D* p, Trajectory* t) {
	Vector2D u,v,dist;
	v = t->direction;
	minus_void(p,&(t->source), &u);

	float pp = dot_product(&u,&v) / dot_product(&v,&v);

	Vector2D proj;
	set(&proj,v.x, v.y);

	normalize_void(&proj);
	mult_constant_void(&proj,pp);
	minus_void(&u,&proj,&dist);

	float mag = magnitude(&dist);

	return mag;
}

__device__ float trajectory_force(Trajectory *t)
{
	int delta = t->n_particulas_estavel - t->n_particulas_atual;
	//	printf("\r\nATUAL = %d \t ESTAVEL = %d", t->n_particulas_atual, t->n_particulas_estavel);
	return (TRAJ_FORCE_LINEAR*delta>0)?TRAJ_FORCE_LINEAR*delta:0;
}

__device__ int current_status(Trajectory* t)
{
	return (t->n_particulas_estavel-t->n_particulas_atual>0)?t->n_particulas_estavel-t->n_particulas_atual:0;
}



__device__ inline void directionFrom(Vector2D *point, Trajectory *t, Vector2D *direction)
{
	Vector2D begin, end;// = clone(t->source);
	set(&begin, t->source.x,t->source.y);
	set(&end, 0.0,0.0);

	sum_void(&(t->source), &(t->direction), &end);

	Vector2D u, v;
	set(&u, point->x - begin.x, point->y - begin.y);
	set(&v, end.x - begin.x, end.y - begin.y);

	float pp = ((u.x * v.x) + (u.y * v.y)) / ((v.x * v.x) + (v.y * v.y));

	Vector2D projuv;
	set(&projuv, pp * v.x, pp * v.y);

	direction->x = u.x - projuv.x;
	direction->y = u.y - projuv.y;


	//	return new_vector(u->x - projuv->x, u->y - projuv->y);
}

__device__ void update_trajectory(Trajectory *t, Particle **p, int nparticle)
{
	t->n_particulas_atual = 0;

	int i=0;
	for (i=0;i<nparticle;i++)
	{
		if (p[i]->status == ALIVE)
		{
			// Find distance from point to line segment (orthogonal)
			float distance_point_line = distance(p[i]->location,t);

			if (distance_point_line<TRAJ_PART_THRESHOLD)
			{
				t->n_particulas_atual++;
				p[i]->current_trajectories++;
			}
		}
	}
}

__device__ int trajectory_stable(Trajectory *t)
{
	if (t->n_particulas_atual>=t->n_particulas_estavel) return TRUE;
	else
	{
		//		printf("\r\n %f \t %d \t %d",t->direction->x,t->n_particulas_estavel,t->n_particulas_atual);
		return FALSE;
	}
}

// Gravitacao universal
__device__ inline void resultant(Trajectory *t, Particle* p, Vector2D *resultant)
{
	directionFrom(p->location, t, resultant);
	//	normalize(ortonormal);		//TODO aparentemente nao influencia movimento das particulas

	// Find distance from point to line segment (orthogonal)
	float distance_point_line = distance(p->location,t);
	// Calculate force according to F = m1*m2/r²
	float mass_trajectory = trajectory_force(t);
	float mass_particle   = p->mass;
	float force = 0;
	if (distance_point_line>0.01)
		force = (mass_trajectory * mass_particle) / (distance_point_line*distance_point_line);
	force/=10;

	mult_constant_void(resultant,force);
}
