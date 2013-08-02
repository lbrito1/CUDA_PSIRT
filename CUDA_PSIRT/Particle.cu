/*
 * Particle.h
 *
 *  Created on: Mar 25, 2013
 *      Author: Leo
 */

#define DEAD 	 0
#define ALIVE 	 1
#define CHECKING 2


#define PARTICLE_MASS 0.01

#include "Vector2D.cu"

// DATA STRUCTURES
typedef struct {
	Vector2D location;
	int current_trajectories;
	int traj_1, traj_2, traj_3;
	int status;
} Particle;

// HOST FUNCTIONS
Particle* new_particle();

// DEVICE FUNCTIONS
__host__ __device__ void update_particle(Particle* particle, Vector2D* resultant_force);

double particle_mass = 0.01;

Particle* new_particle()
{
	Particle* part = (Particle*)malloc(sizeof(Particle));
	Vector2D* loc = new_vector(0.0,0.0);
	part->location = *loc;
	part->current_trajectories = 0;
	part->status = ALIVE;
	return part;
}

__host__ __device__ void update_particle(Particle* particle, Vector2D* resultant_force)
{
	mult_constant_void(resultant_force, PARTICLE_MASS);
	sum_void(&particle->location, resultant_force, &particle->location);
}
