#include "Reconstruction.c"
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <windows.h>
#include <time.h>
#include <GL/glut.h>

#include <cuda_gl_interop.h>

inline void GPUassert(cudaError_t code, char * file, int line, bool Abort=true)
{
    if (code != 0) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code),file,line);
        if (Abort) exit(code);
    }       
}

#define GPUerrchk(ans) { GPUassert((ans), __FILE__, __LINE__); }


//gambiarra
//#define DEBUG_OPENGL
#define DEBUG_PRINT

void prep_psirt();
void display();
void test_psirt();

__global__ void psirt_kernel(PSIRT* psirt, Projection** dev_proj);

GLuint positionsVBO;
struct cudaGraphicsResource *positionsVBO_CUDA;

PSIRT *debug_host_psirt;
PSIRT *dev_psirt;

Projection**	dev_projections;
Particle**		dev_particles;

// ---------------------------
// *** OPENGL ***
// Desenhar configuracao (trajetorias)
// ---------------------------
void opengl_draw_configuration_lines(PSIRT* psirt)
{
 glClear(GL_COLOR_BUFFER_BIT);



  glBegin(GL_LINES);
  int i, j;
	for (i = 0; i < psirt->n_projections; i++)
	{
		for (j = 0; j < psirt->n_trajectories; j++)
		{
			Trajectory* t = psirt->projections[i]->lista_trajetorias[j];

			Vector2D begin, end, d;
			sum_void(&(t->source), &(t->direction), &begin);
			d.x = t->direction.x;
			d.y = t->direction.y;
	
			mult_constant_void(&d, -1);
			sum_void(&(t->source), &d, &end);

			glColor3f(1.0, 1.0, 1.0);
			glVertex2f(begin.x, begin.y);
			glVertex2f(end.x, end.y);
			
		}
	}
	glEnd();
}

// ---------------------------
// *** OPENGL ***
// Desenhar particulas
// ---------------------------
void opengl_draw_particles(PSIRT* psirt)
{
	int i=0;
	glPointSize(7.0);
	glBegin(GL_POINTS);
	for (i = 0; i < psirt->n_particles; i++) {
		if (psirt->particles[i]->status != DEAD) {
			glColor3f(1.0, 0.0, 0.0);
			glVertex2f(psirt->particles[i]->location->x, psirt->particles[i]->location->y);
		} else {
			glColor3f(0.0, 1.0, 0.0);
			glVertex2f(psirt->particles[i]->location->x, psirt->particles[i]->location->y);
		}
		//printf("\r\n%d: \t%f,%f",i,psirt->particles[i]->location->x, psirt->particles[i]->location->y);
	}

	glEnd();
	glFlush();  // Render now
}

void opengl_draw()
{
	

	// Copia dev->host
	cudaMemcpy(debug_host_psirt, dev_psirt, sizeof(PSIRT), cudaMemcpyDeviceToHost);
	//printf("\r\nAAA->%d",debug_host_psirt->optim_max_iterations);
	opengl_draw_configuration_lines(debug_host_psirt);
	opengl_draw_particles(debug_host_psirt);

	
}
void update()
{
	// Executa uma iteração
//	psirt_kernel<<<1,1>>>(dev_psirt);
	
	// Desenha
	glutPostRedisplay();
}

void keyboard_handler (unsigned char key, int x, int y)
{
	if (key == 27) exit(0);	//ESC = exit
}


void init_opengl(int argc, char* argv[])
{
	glutInit(&argc, argv);
	glutCreateWindow("PSIRT_CUDA");
	glutReshapeWindow(RES_X, RES_Y);
	glutDisplayFunc(opengl_draw);
	glutIdleFunc(update);
	glutKeyboardFunc(keyboard_handler);
	glutMainLoop();
}











__global__ void ts(Vector2D* l)
{
	printf("\r\nx=%f",l[0].x);
}

__global__ void test(Trajectory* t)
{
	printf("\r\nPart = %d",t[0].n_particulas_atual);
}

void test_psirt(PSIRT* host_psirt)
{
	int n_proj = host_psirt->n_projections;
	int n_traj = host_psirt->n_trajectories;
	int n_ttl_traj = n_proj * n_traj;
	
	int ttl_traj_len = n_proj * n_traj;
	size_t ttl_traj_size = sizeof(Trajectory) * ttl_traj_len;

	Trajectory* traj;
	GPUerrchk(cudaMalloc((void**)&traj, ttl_traj_size));

	int i,j,k;
	
	for (i=0,k=0; i<n_proj; i++) 
	{
		Projection *p = host_psirt->projections[i];

		for (j=0; j< n_traj; j++,k++)
		{
			Trajectory t = *p->lista_trajetorias[j];
			
			GPUerrchk(cudaMemcpy(&(traj[k]), &t, sizeof(Trajectory), cudaMemcpyHostToDevice));
		}
	}
	test<<<1,1>>>(traj);
	
	
}














void prep_psirt() 
{
	cudaError_t cudaStatus;

	PSIRT* host_psirt = init_psirt();

	
	if (cudaStatus != cudaSuccess) fprintf(stderr, "cudaMalloc failed!");


	  //Sending Side
    char b[sizeof(PSIRT)];
    memcpy(b, &host_psirt, sizeof(PSIRT));

    //Receiving Side
    PSIRT* tmp; //Re-make the struct
	//cudaMalloc((void**)&tmp, sizeof(PSIRT));
	//cudaMemcpy(tmp, b, sizeof(PSIRT), cudaMemcpyHostToDevice);


	//psirt_kernel<<<1,1>>>(tmp, dev_projections);

/*	// 1. INICIALIZAR (HOST)
	PSIRT* host_psirt = init_psirt();

	// 2. COPIAR (HOST->DEV)			---- CHECAR!!!!!!!!!!



	// TEMP --------------
	size_t size_proj = sizeof(host_psirt->n_projections)*sizeof(Projection);
	size_t size_part = sizeof(host_psirt->n_particles  )*sizeof(Particle);

	cudaMalloc((void**)&dev_projections, size_proj );
	cudaMalloc((void**)&dev_particles,	size_part );

	int k = 0;
	for(k=0;k<1;k++)
		cudaMemcpy(dev_projections[k], host_psirt->projections[k], sizeof(Projection), cudaMemcpyHostToDevice);
	
	
	cudaMemcpy(dev_particles,	host_psirt->particles,	 size_part, cudaMemcpyHostToDevice);
	// TEMP --------------



	cudaMalloc((void**)&dev_psirt, sizeof(PSIRT));
	cudaMemcpy(dev_psirt, host_psirt, sizeof(PSIRT), cudaMemcpyHostToDevice);


	//dev_psirt->projections = dev_projections;
	//dev_psirt->particles	= dev_particles;


	// 3. EXECUTAR (DEV)
	psirt_kernel<<<1,1>>>(dev_psirt, dev_projections);
	cudaDeviceSynchronize();

	// 4. COPIAR (DEV->HOST)
	cudaMemcpy(host_psirt, dev_psirt, sizeof(PSIRT), cudaMemcpyDeviceToHost);
	cudaFree(dev_psirt);

	// 5. RECONSTRUCAO (HOST)
	draw_projection_bitmap(host_psirt);
	draw_reconstruction_bitmap(host_psirt);
	
	free(host_psirt);*/
}



__global__ void psirt_kernel(PSIRT* psirt, Projection** dev_proj)
{
	//int q = 0;
//	printf("%d",q);
//	for(q=0;q<100;q++){
	//	printf("\r\n\r\n%d",q);
	// ---------------------------
	// *** ATUALIZAR POSICOES DAS PARTICULAS ***
	// ---------------------------
	// ---------------- update_particles(psirt);

	int i=0,j=0,k=0;
	Vector2D resultant_force;
	//printf("\r\nProj = %d",dev_proj[0]->n_traj);
	//printf("\r\nProj = %d",dev_projections[0]->n_traj);
	//psirt->n_particles = 2; // funciona
}



int main(int argc, char* argv[])
{
	cudaError_t cudaStatus;

		PSIRT* host_psirt = init_psirt();
	
	#ifdef DEBUG_OPENGL
		// 1. INICIALIZAR (HOST)
		debug_host_psirt = init_psirt();

		// 2. COPIAR (HOST->DEV)
		cudaMalloc((void**)&dev_psirt, sizeof(PSIRT));
		cudaMemcpy(dev_psirt, debug_host_psirt, sizeof(PSIRT), cudaMemcpyHostToDevice);

		// 3. EXECUTAR / DESENHAR
		init_opengl(argc, argv);
	#endif

	#ifndef DEBUG_OPENGL
		cudaError_t cuda_status = cudaSetDevice(0);
		if (cuda_status != cudaSuccess) printf("\r\n cuInit failed");

		//prep_psirt(); // CUDA

		test_psirt(host_psirt);
	#endif

	

    
	//printf("\r\nOK!");
	//scanf("%d");

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }
	


	draw_projection_bitmap(host_psirt);
	draw_reconstruction_bitmap(host_psirt);
	
	free(host_psirt);

    return 0;
}