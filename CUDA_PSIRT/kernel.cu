#include "Reconstruction.c"
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <windows.h>
#include <time.h>
#include <GL/glut.h>

#include <cuda_gl_interop.h>

#define GPUerrchk(ans) { GPUassert((ans), __FILE__, __LINE__); }


//gambiarra
//#define DEBUG_OPENGL
//#define DEBUG_PRINT

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
			Trajectory* t = &(psirt->trajectories[(i*psirt->n_projections)+j]);

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
		if (psirt->particles[i].status != DEAD) {
			glColor3f(1.0, 0.0, 0.0);
			glVertex2f(psirt->particles[i].location.x, psirt->particles[i].location.y);
		} else {
			glColor3f(0.0, 1.0, 0.0);
			glVertex2f(psirt->particles[i].location.x, psirt->particles[i].location.y);
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

__global__ void test(Trajectory* t, Particle* p, int* dev_params, PSIRT* dev_psirt)
{
	//printf("\r\n===========\r\nSTARTUP CUDA PSIRT\r\n===========\r\nPARAMS:");
	//printf("\t(#PROJ)\t(#TRAJ)\t(NPART)\r\n\t%d\t%d\t%d",dev_params[0], dev_params[1], dev_params[2]);

	dev_psirt->particles = p;
	dev_psirt->trajectories = t;
	
	dev_psirt->n_projections = dev_params[0];
	dev_psirt->n_trajectories = dev_params[1];
	dev_psirt->n_particles = dev_params[2];

	dev_psirt->is_optimized = 0;
	dev_psirt->is_optimizing_dirty_particle = 0;
	dev_psirt->optim_is_ranked = 0;
	dev_psirt->optim_curr_part = 0;
	dev_psirt->optim_curr_iteration = 0;
	dev_psirt->optim_max_iterations = 100;

	int ttl_trajs = dev_psirt->n_trajectories * dev_psirt->n_projections;

	int is_optimizing_dirty_particle = 0;


	printf("\r\n===========\r\nSTARTUP CUDA PSIRT\r\n===========\r\nPARAMS:");
	printf("\t(#PROJ)\t(#TRAJ)\t(NPART)\r\n\t%d\t%d\t%d\r\n\r\n",dev_psirt->n_projections, dev_psirt->n_trajectories, dev_psirt->n_particles);



	int done = 0;
	int lim = 0;


	while (!done&(++lim<1000))
	{
			// ---------------------------
		// *** ATUALIZAR POSICOES DAS PARTICULAS ***
		// ---------------------------
		update_particles(dev_psirt);

		// ---------------------------
		// *** CALCULO DE TRAJETORIAS SATISFEITAS ***
		// ---------------------------
		int i=0,j=0;
		for (i=0;i<dev_psirt->n_particles;i++) dev_psirt->particles[i].current_trajectories = 0; 	// zera #traj de cada particula
		for (i=0;i<ttl_trajs; i++) 											// calcula #traj de cada particula
			update_trajectory(dev_psirt->trajectories[i], dev_psirt->particles, dev_psirt->n_particles);

		// ---------------------------
		// *** OTIMIZACAO E CONVERGENCIA ***
		// ---------------------------
		optimization_check(dev_psirt);													// otimizacao (continuar)

		if (has_converged(dev_psirt->trajectories,dev_psirt->n_projections*dev_psirt->n_trajectories))							// convergiu
		{
			if (dev_psirt->optim_curr_part < dev_psirt->n_particles) optimize(dev_psirt);						// otimizacao (comecar)
			else done = 1;	// DONE 
		//	done = 1;
		}
//	return 0;	// NOT DONE
	}
}

void test_psirt(PSIRT* host_psirt)
{

	// 1: COPIAR PROJEÇÕES/TRAJETÓRIAS
	
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
		for (j=0; j< n_traj; j++,k++)
		{
			Trajectory t = (host_psirt->trajectories[(i*host_psirt->n_projections)+j]);
			GPUerrchk(cudaMemcpy(&(traj[k]), &t, sizeof(Trajectory), cudaMemcpyHostToDevice));
		}
	}


	// 2. COPIAR PARTÍCULAS

	int n_part = host_psirt->n_particles;
	int ttl_part_size = sizeof(Particle) * n_part;

	Particle *part;
	GPUerrchk(cudaMalloc((void**)&part, ttl_part_size));

	for (i=0; i<n_part; i++)
	{
		Particle hp = host_psirt->particles[i];
		GPUerrchk(cudaMemcpy(&(part[i]), &hp, sizeof(Particle), cudaMemcpyHostToDevice));
	}

	// 3. DEMAIS PARAMETROS
	int params[] = {n_proj, n_traj, n_part};
	int *dev_params;
	GPUerrchk(cudaMalloc((void**)&dev_params, sizeof(int)*3));
	GPUerrchk(cudaMemcpy(dev_params,params,sizeof(int)*3,cudaMemcpyHostToDevice));

	
	// 3. REMONTAR PARAMETROS
	PSIRT* dev_psirt;
	Projection* dev_proj;
	GPUerrchk(cudaMalloc((void**)&dev_psirt, sizeof(PSIRT)));

	test<<<1,1>>>(traj, part, dev_params, dev_psirt);

	// 4. COPIAR DE VOLTA
	Particle *host_plist = host_psirt->particles;
	
		
	GPUerrchk(cudaMemcpy( host_plist, part, sizeof(Particle) * n_part, cudaMemcpyDeviceToHost));
	

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