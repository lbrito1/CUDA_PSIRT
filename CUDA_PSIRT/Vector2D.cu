/*
 *
 *  Created on: Mar 24, 2013
 *      Author: Leo
 */

#include "cuda.h"
#include "cuda_runtime.h"
#include <math.h>
#define PI 3.1415926536
#define DEGREE_TO_RAD PI/180.0
#define RAD_TO_DEGREE 180.0/PI

#if defined(__CUDACC__) // NVCC
   #define MY_ALIGN(n) __align__(n)
#elif defined(__GNUC__) // GCC
  #define MY_ALIGN(n) __attribute__((aligned(n)))
#elif defined(_MSC_VER) // MSVC
  #define MY_ALIGN(n) __declspec(align(n))
#else
  #error "Please provide a definition for MY_ALIGN macro for your host compiler!"
#endif

// DATA STRUCTURES
//MY_ALIGN(8)
typedef struct { float x,y; } Vector2D;

// HOST FUNCTIONS
Vector2D* new_vector(float x, float y); 
Vector2D* clone(Vector2D* a);

// DEVICE FUNCTIONS
__host__ __device__ void set_vector(Vector2D* a, float x, float y);
__host__ __device__ void sum_void(Vector2D* a, Vector2D* b, Vector2D* result);
__host__ __device__ void minus_void(Vector2D* a, Vector2D* b, Vector2D* result);
__device__ float dot_product(Vector2D* a, Vector2D* b);
__device__ float magnitude(Vector2D* a);
__device__ void mult_constant_void(Vector2D* a, float k);
__device__ void normalize(Vector2D* a);
__host__ __device__ void set(Vector2D* v, float x, float y);
__device__ void normalize_void(Vector2D* a);
__device__ void copyTo(Vector2D* a, Vector2D* b);
__device__ void RotateCounterClockWise(Vector2D* a, float angle);
__device__ void RotateClockWise(Vector2D* a, float angle);
__host__ __device__ float vector_vector_distance(Vector2D* a, Vector2D* b);
__device__ void lerp(Vector2D* a, Vector2D* b, float t, Vector2D* c);
__device__ float distance_point_line(Vector2D* point, Vector2D* line_a, Vector2D* line_b);


__host__ __device__ void set_vector(Vector2D* a, float x, float y)
{
	a->x=x;a->y=y;
}

__host__ __device__ void sum_void(Vector2D* a, Vector2D* b, Vector2D* result)
{
	result->x = a->x+b->x;
	result->y = a->y+b->y;
}

__host__ __device__ void minus_void(Vector2D* a, Vector2D* b, Vector2D* result)
{
	result->x = a->x-b->x;
	result->y = a->y-b->y;
}

__device__ float dot_product(Vector2D* a, Vector2D* b)
{
	return (a->x*b->x + a->y*b->y);
}


__host__ __device__ float magnitude(Vector2D* a)
{
	return sqrt((a->x * a->x)+(a->y * a->y));
}

Vector2D* new_vector(float x, float y) {
	Vector2D* a = (Vector2D*) malloc(sizeof(Vector2D));	//TODO 
	a->x=x;
	a->y=y;
	return a;
}

__host__ __device__ void mult_constant_void(Vector2D* a, float k)
{
	a->x=a->x*k;
	a->y=a->y*k;
}


__host__ __device__ void normalize(Vector2D* a)
{
	mult_constant_void(a,1/magnitude(a));
}

__device__ void normalize_void(Vector2D* a)
{
	float constant = 1/magnitude(a);
	mult_constant_void(a,constant);
}

Vector2D* clone(Vector2D* a)
{
	Vector2D* cloned = (Vector2D*) malloc(sizeof(Vector2D));
	cloned->x = a->x;
	cloned->y = a->y;
	return cloned;
}

__device__ void copyTo(Vector2D* a, Vector2D* b)
{
	b->x = a->x;
	b->y = a->y;
}

__host__ __device__ void RotateCounterClockWise(Vector2D* a, float angle)
{
    float angleRad = angle *DEGREE_TO_RAD;
    float x = a->x * cos(angleRad) - a->y * sin(angleRad);
    float y = a->x * sin(angleRad) + a->y *cos(angleRad);
    a->x = x;
	a->y = y;   
}

__host__ __device__ void set(Vector2D* v, float x, float y)
{
	v->x=x;
	v->y=y;
}


__host__ __device__ void RotateClockWise(Vector2D* a, float angle)
{
	RotateCounterClockWise(a, -angle);
}


__host__ __device__ float vector_vector_distance(Vector2D* a, Vector2D* b)
{
  Vector2D c; 
  minus_void(a,b,&c);
  return magnitude(&c);
}


__host__ __device__ void lerp(Vector2D* a, Vector2D* b, float t, Vector2D* c)
{
	c->x = a->x * t + b->x * (1-t);
	c->y = a->y * t + b->y * (1-t);
	
}

__device__ float distance_point_line(Vector2D* point, Vector2D* line_a, Vector2D* line_b)
{
	float x0,y0,x1,y1,x2,y2;
	x0 = point->x; y0 = point->y;
	x1 = line_a->x; y1 = line_a->y;
	x2 = line_b->x; y2 = line_b->y;

	return ((fabs(((x2-x1)*(y1-y0))-((x1-x0)*(y2-y1)))) / sqrt(pow((x2-x1),2))+pow((y2-y1),2));
}
