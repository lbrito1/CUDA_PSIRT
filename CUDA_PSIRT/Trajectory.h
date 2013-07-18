/*
 * Trajectory.h
 *
 *  Created on: Apr 1, 2013
 *      Author: Leo
 */

#pragma once

#ifndef TRAJECTORY_H_
#define TRAJECTORY_H_


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

#include "Vector2D.h"
#include "Particle.h"

typedef struct {
	Vector2D* source, *direction;
	int n_particulas_estavel;
	int n_particulas_atual;
} Trajectory;

int trajectory_stable(Trajectory *t);

Vector2D* trajectory_intersection(Trajectory *t1, Trajectory *t2);

Trajectory* new_trajectory(Vector2D* s, Vector2D* d, int part_est);

float distance(Vector2D* p, Trajectory* t);

Vector2D* projection(Vector2D* p, Trajectory* t);

int current_status(Trajectory* t);

double trajectory_force(Trajectory* t);

inline void resultant(Trajectory *t, Particle* p, Vector2D *resultant);

void update_trajectory(Trajectory *t, Particle **p, int nparticle);

inline void directionFrom(Vector2D *point, Trajectory *t, Vector2D *direction);

#endif /* TRAJECTORY_H_ */
