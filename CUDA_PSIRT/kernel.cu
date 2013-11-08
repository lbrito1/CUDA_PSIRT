#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <windows.h>
#include <time.h>
#include <GL/glut.h>
#include <math.h>

#define RES_X 16
#define RES_Y RES_X

#define MAT_DIM RES_X

#define MAT_SIZE MAT_DIM*MAT_DIM

#define RES_FACTOR 10

// Matriz MAT_DIM x MAT_DIM
typedef int* MTraj;

typedef MTraj* MTrajArray;

int n_traj = 1;

int belongs(float x, float y, float a, float b) { return (y==a*x + b) ? true : false; }

int get_index(int x, int y) { return x*MAT_DIM+y <= MAT_SIZE ? x*MAT_DIM+y : -1; }

void prep_MTraj(int *MT, float a, float b)
{
	// y = ax + b
	for (int i=0; i<MAT_DIM; i++) 
	{
		for (int j=0; j<MAT_DIM; j++)
		{
			if (belongs(i,j,a,b)) MT[get_index(i,j)] = 0;
			else 
			{
				float dx = abs(i-((j-b)/a));
				float dy = abs(j-a*i-b);
				MT[get_index(i,j)] = (int)sqrt(dx*dx + dy*dy);
			}
			printf(" %d", MT[get_index(i,j)]);
		}

		printf("\r\n");
	}
}


void test()
{
	int *MT = (int*) malloc(sizeof(int) * MAT_SIZE);
	float a = 1.0f;
	float b = 0.0f;

	prep_MTraj(MT, a, b);


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

float *dev_x;
float *host_x;

__global__ void CUDA_work(float* x)
{
	if (*x<1.0f) *x += .1;
	else *x = 0.0f;
}

int color = 0;

void opengl_draw()
{
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT);
	
	glBegin(GL_POLYGON);

	glVertex2f(*host_x,-*host_x);
	glVertex2f(0.6,0.6);
	glVertex2f(0.6,0.2);

	glEnd();
	glFlush();	
	Sleep(50);
}
void update()
{
	float h = *host_x;
	GPUerrchk(cudaMemcpy(dev_x, host_x, sizeof(float), cudaMemcpyHostToDevice));
	CUDA_work<<<1,1>>>(dev_x);
	GPUerrchk(cudaMemcpy(host_x, dev_x, sizeof(float), cudaMemcpyDeviceToHost));
	h = *host_x;
	printf("\r\n %f",*host_x);
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
	host_x = (float*) malloc(sizeof(float));
	*host_x = 0.0f;
	cudaError_t cudaStatus;
	GPUerrchk(cudaMalloc((void**)&dev_x, sizeof(float)));
	


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
