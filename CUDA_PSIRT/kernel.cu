#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <windows.h>
#include <time.h>
#include <GL/glut.h>
#include <math.h>

#define PI 3.14159265
#define RAD(X) ((X*PI/180.0))


#define RES_X 160
#define RES_Y RES_X

#define MAT_DIM RES_X

#define MAT_SIZE MAT_DIM*MAT_DIM

#define RES_FACTOR 10

// Matriz MAT_DIM x MAT_DIM
typedef int* MTraj;

MTraj host_MTraj;
MTraj dev_MTraj;

int n_traj = 1;

int belongs(double x, double y, double a, double b) { return (y==a*x + b) ? true : false; }

inline int get_index(int x, int y) { return x*MAT_DIM+y <= MAT_SIZE ? x*MAT_DIM+y : -1; }
inline int get_MT_offset(int i) { return i*MAT_SIZE; }
inline int get_MT_idx(int ofs, int x, int y) { return get_MT_offset(ofs)+get_index(x,y); }

inline double to_GL_coord(int x) { return (x/(double)MAT_DIM) - 1.0f; }

void prep_MT_cells(int *MT, int offset, double a, double b)
{
	
	//printf("\r\nPreparando matriz id#%d, eq y=%f*x+%f\r\n\------------r\n",get_MT_offset(offset), a, b);
	// y = ax + b
	for (int j=0; j<MAT_DIM; j++) 
	{
		for (int i=0; i<MAT_DIM; i++)
		{
			//printf("\r\nPreparando celula # %d,%d: ",i,j);

			if (belongs(i,j,a,b)) MT[get_MT_idx(offset,i,j)] = 0;
			else 
			{
				double dx = a>0.0001 ? abs(i-((j-b)/a)) : 0.0f; // casos perpendiculares (dist=0)
				double dy = abs(j-a*i-b);

				MT[get_MT_idx(offset,i,j)] = (int)sqrt(dx*dx + dy*dy);
			}
		//	printf(" %d", MT[get_MT_idx(offset,i,j)]);
		}

		//printf("\r\n");
	}
}

// Primeira trajetória da projeção (intersecta todas as outras
// diretoras das outras projs em x = MAT_DIM/2, y = MAT_DIM/2
inline void get_eq_traj_director(double in_ang, double *out_a, double *out_b) 
{
	*out_a = tan(RAD(in_ang));
	*out_b = (MAT_DIM*(1-*out_a))/2;
}

// config m x n, e.g. 3x7
//in: config
//out: array de arrays (matrizes) de inteiros 
int* prep_MT_from_config(int m, int n)
{
	int* root = (int*) malloc(sizeof(int)*m*n*MAT_SIZE);

	double ang = 180.0/3;

	int d_traj = max(2, MAT_SIZE/8);	

	// Projections
	for (int i=0, offset=0; i<m; i++) 
	{
		// Trajectories
		for (int j=0; j<n; j++,offset++) 
		{
			double a, b;
			get_eq_traj_director(ang*j, &a, &b);
			prep_MT_cells(root, offset, a, b);
		}
	}

	return root;
}



void test()
{
	n_traj = 21;
	host_MTraj = prep_MT_from_config(3,7);

	/*int* root = (int*) malloc(sizeof(int)*MAT_SIZE*2);

	double ang = 120.0f;

	double a, b;
	
	get_eq_traj_director(60.0f, &a, &b);
	prep_MT_cells(root, 0, a, b);

	get_eq_traj_director(ang, &a, &b);
	prep_MT_cells(root, 1, a, b);*/
}


#include <cuda_gl_interop.h>

inline void GPUassert(cudaError_t code, char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %dn", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}
#define GPUerrchk(ans) { GPUassert((ans), __FILE__, __LINE__); }

GLuint positionsVBO;
struct cudaGraphicsResource *positionsVBO_CUDA;

double *dev_x;
double *host_x;

__global__ void CUDA_work(double* x)
{
	if (*x<1.0f) *x += .1;
	else *x = 0.0f;
}

int color = 0;

void opengl_draw()
{
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT);
	
/*	glBegin(GL_POLYGON);

	glVertex2f(*host_x,-*host_x);
	glVertex2f(0.6,0.6);
	glVertex2f(0.6,0.2);

	glEnd();*/

	



	glPointSize(1.0);
	glBegin(GL_POINTS);

	glColor3f(1.0f,1.0f,1.0f);
	glVertex2d(0.0,0.0);

	for (int k=0; k<n_traj; k++) 
	{

		for (int i=0;i<MAT_DIM; i++) 
		{
			for (int j=0;j<MAT_DIM; j++) 
			{
				if ((host_MTraj[get_MT_idx(k,i,j)]) < .0001) 
				{
					//printf("\r\n ij = %d, %d >>> coord= %f, %f",i,j,to_GL_coord(i),to_GL_coord(j));
					glVertex2d(to_GL_coord(i), to_GL_coord(j));
				}

			}
		}
	}

	glEnd();


	glFlush();	
	Sleep(50);
}
void update()
{
	double h = *host_x;
	GPUerrchk(cudaMemcpy(dev_x, host_x, sizeof(double), cudaMemcpyHostToDevice));
	CUDA_work<<<1,1>>>(dev_x);
	GPUerrchk(cudaMemcpy(host_x, dev_x, sizeof(double), cudaMemcpyDeviceToHost));
	h = *host_x;
	//printf("\r\n %f",*host_x);
	GPUerrchk(cudaDeviceSynchronize());
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






int main(int argc, char* argv[])
{
	host_x = (double*) malloc(sizeof(double));
	*host_x = 0.0f;
	cudaError_t cudaStatus;
	GPUerrchk(cudaMalloc((void**)&dev_x, sizeof(double)));
	


	test();


	// 3. EXECUTAR / DESENHAR
	init_opengl(argc, argv);

    cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}
