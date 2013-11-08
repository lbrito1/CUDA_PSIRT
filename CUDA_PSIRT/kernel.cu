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

GLuint positionsVBO;
struct cudaGraphicsResource *positionsVBO_CUDA;


int color = 0;

void opengl_draw()
{
	
	glBegin(GL_POLYGON);

	glVertex2f(0.2,0.2);
	glVertex2f(0.6,0.6);
	glVertex2f(0.6,0.2);

	glEnd();
	glFlush();	
}
void update()
{

	glutPostRedisplay();
}

void keyboard_handler (unsigned char key, int x, int y)
{
	if (key == 27) exit(0);	//ESC = exit
	else if (key == 27) color = color == 0?1:0;
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
	cudaError_t cudaStatus;

	// 3. EXECUTAR / DESENHAR
	init_opengl(argc, argv);

    cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}