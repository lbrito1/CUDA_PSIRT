/*
 * Particle.h
 *
 *  Created on: Mar 25, 2013
 *      Author: Leo
 */

#define DEAD 	 0
#define ALIVE 	 1
#define CHECKING 2

#include "Vector2D.cu"

// DATA STRUCTURES
typedef struct {
	Vector2D* location;
	Vector2D* speed;
	Vector2D* acceleration;
	double mass;
	int current_trajectories;
	int status;
} Particle;

// HOST FUNCTIONS
Particle* new_particle();

// DEVICE FUNCTIONS
__device__ void update_particle(Particle* particle, Vector2D* resultant_force);

double particle_mass = 0.01;

Particle* new_particle()
{
	Particle* part = (Particle*)malloc(sizeof(Particle));
	Vector2D* loc = new_vector(0.0,0.0);
	Vector2D* spd = new_vector(0.0,0.0);
	Vector2D* acc = new_vector(0.0,0.0);
	part->location = loc;
	part->speed = spd;
	part->acceleration = acc;
	part->mass = particle_mass;
	part->current_trajectories = 0;
	part->status = ALIVE;
	return part;
}

__device__ void update_particle(Particle* particle, Vector2D* resultant_force)
{
	double particle_mass = 0.01;
	// actual acceleration delta
	if (particle->mass<.00001)
	{
		resultant_force->x=0;
		resultant_force->y=0;
	}
	else
	{
		mult_constant_void(resultant_force, particle_mass);	// TODO gambiarra física
	}
	// update acceleration with new delta
	particle->acceleration->x = resultant_force->x;
	particle->acceleration->y = resultant_force->y;

	particle->speed->x=0.0;
	particle->speed->y=0.0;

	// update speed by summing curent speed vector with acceleration vector
	sum_void(particle->speed, particle->acceleration, particle->speed);

	// update location
	sum_void(particle->location, particle->speed, particle->location);
}