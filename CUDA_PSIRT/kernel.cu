#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <windows.h>
#include <time.h>
#include <stdlib.h>
#include <GL/glut.h>
#include <math.h>
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


#define PI 3.14159265
#define RAD(X) ((X*PI/180.0))

#define RES_X 160
#define RES_Y RES_X

#define MAT_DIM RES_X

#define MAT_SIZE MAT_DIM*MAT_DIM

#define RES_FACTOR 10

#define FATOR_SEP 1

#define FATOR_PROX 0.15


int dbg_showvec = 0;
int dbg_trajid = 0;
int dbg_CUDA_active = 0;

struct hfloat2 { float x; float y; };


typedef int* MTraj;	// Matriz MAT_DIM x MAT_DIM
MTraj host_MTraj;
MTraj dev_MTraj;
int n_traj = 1;
int n_proj;

typedef int* MPart;	// Matriz MAT_DIM x MAT_DIM, cada elem>0 = 1 particula, id=1..MAX_INT
MPart host_MPart;
MPart dev_MPart;
int n_part = 10;

typedef float2* MVector; // Matriz MAT_DIM x MAT_DIM, cada elem = vetor unidade força naquele ponto	
MVector host_MV;
MVector dev_MV;

int *host_MT_Sum;

inline int belongs(double x, double y, double a, double b) { return (y==a*x + b) ? true : false; }

inline int get_index(int x, int y) { return x*MAT_DIM+y <= MAT_SIZE ? x*MAT_DIM+y : -1; }
__host__ __device__ inline int get_MT_offset(int i) { return i*MAT_SIZE; }
__host__ __device__ inline int M_idx(int ofs, int x, int y) { return get_MT_offset(ofs)+get_index(x,y); }

inline float2 rotate_vector(float2 v, float ang) 
{
	float2 nv;
	nv.x = v.x*cos(ang) - v.y*sin(ang);
	nv.y = v.x*sin(ang) - v.y*cos(ang);
	return nv;
}

inline int fator_separacao_trajs(int ntraj_per_proj) { return (int) ((double)MAT_DIM/((double)FATOR_SEP*(double)ntraj_per_proj)); }

//retorna se a partícula pertence à trajetória distante dist da partícula. (saída: distancia se pertence ou 0 para desconsiderar)
inline int part_is_in_traj(int dist, int ntraj_per_proj) { return dist <= (int) (ntraj_per_proj*((double)FATOR_PROX)) ? dist : 0 ;  };

inline double to_GL_coord(int x) { return (2*x/(double)MAT_DIM) - 1.0f; }

inline float sum_angle(float angle, float sum) 
{  
	float x = angle+sum;
	if (x>360.0f) return x-360.0f;
	else if (x<0.0f) return 360.0f - abs(x);
	else return x;
}

inline hfloat2 get_MT_vectors(float angle)
{
	hfloat2 vectors;
	vectors.x = sum_angle(angle, 90.0f);
	vectors.y = sum_angle(angle, -90.0f);
	return vectors;
}

double *dev_x;
double *host_x;



__global__ void CUDA_APSIRT(MTraj MT, MPart MP, MVector MV, int* np_stb, int* np_cur, int* traj_stb, int* cfg_nproj, int* cfg_ntraj, int* dev_npart)
{
	// 0. Setup CUDA, zerar ntraj estáveis
	int tid_x = blockIdx.x * blockDim.x + threadIdx.x;	
	int tid_y = blockIdx.y * blockDim.y + threadIdx.y;	
	int tid_z = blockIdx.z;

	//*traj_stb = 0;
	///!!!!!!!!!!
	*traj_stb = 2;
	if(*traj_stb>0)printf("!!!!!!!!!!!!!!!");
	// 1. Calcular força resultante
	/*int dist =  MT[M_idx(tid_z, tid_x, tid_y)];
	int deltapart = np_stb[tid_z]-np_cur[tid_z];

	double resultant = deltapart/dist*dist;

	int fr_x = (int) (MV[tid_z].x * resultant);
	int fr_y = (int) (MV[tid_z].y * resultant);

	// 2. Atualizar posição da partícula					// PERIGO CONCORRENCIA/ USAR SEC ATOMICA
	if (MP[M_idx(0,tid_x,tid_y)] > 0) 
	{
		atomicDec((unsigned int*) &MP[M_idx(0,tid_x,tid_y)], 0);
		atomicInc((unsigned int*) &MP[M_idx(0,fr_x, fr_y) ], 0);
	}*/
	
	// 3. Atualizar trajetórias
	/*int qtd_parts = MP[M_idx(0, tid_x, tid_y)];
	int dist = MT[M_idx(tid_z, tid_x, tid_y)];

	if (qtd_parts>0 && part_is_in_traj(dist, cfg_ntraj)) atomicAdd(&np_cur[tid_z], qtd_parts);

	// 4. Checar convergência
	if (*np_cur >= *np_stb) atomicInc(traj_stb,0);*/
}


inline void CUDA_COPY_int(int* src, int* dst, int length, cudaMemcpyKind kind) 
{
	GPUerrchk(cudaMemcpy(dst, src, sizeof(int)*length, kind));
}

__global__ void CUDA_work(double* x)
{
	if (*x<1.0f) *x += .1;
	else *x = 0.0f;
}
void APSIRT_main_loop(MTraj dev_MT, MPart dev_MP, MPart host_MP, MVector dev_MV, int* dev_np_stb, int* dev_np_cur, int* dev_traj_stb, int* dev_nproj, int* dev_ntraj, int* dev_npart, int host_nttltraj, int *host_traj_stb)
{
	size_t blocks = ceilf( (int)(MAT_DIM) / 16.0f );
	dim3 gridDim( blocks, blocks, host_nttltraj );
	size_t threads = ceilf( (int)(MAT_DIM) / (float)blocks );
	dim3 blockDim( threads, threads, 1 );
 

	CUDA_APSIRT<<< gridDim, blockDim >>>( dev_MT, dev_MP, dev_MV, dev_np_stb, dev_np_cur, dev_traj_stb, dev_nproj, dev_ntraj, dev_npart );
	
	cudaDeviceSynchronize();
	CUDA_COPY_int(dev_traj_stb, host_traj_stb, 1, cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	printf("\r\n>%d",*host_traj_stb);

//CUDA_COPY_int(dev_MP, host_MP, MAT_SIZE, cudaMemcpyDeviceToHost);

	
	double h = *host_x;
	GPUerrchk(cudaMemcpy(dev_x, host_x, sizeof(double), cudaMemcpyHostToDevice));
	CUDA_work<<<1,1>>>(dev_x);
	GPUerrchk(cudaMemcpy(host_x, dev_x, sizeof(double), cudaMemcpyDeviceToHost));
	h = *host_x;
	printf("\r\n %f",*host_x);
}




void prep_MT_cells(int *MT, int offset, double a, double b)
{
	
	//printf("\r\nPreparando matriz id#%d, eq y=%f*x+%f\r\n\------------r\n",get_MT_offset(offset), a, b);
	// y = ax + b
	for (int j=0; j<MAT_DIM; j++) 
	{
		for (int i=0; i<MAT_DIM; i++)
		{
			//printf("\r\nPreparando celula # %d,%d: ",i,j);

			if (belongs(i,j,a,b)) MT[M_idx(offset,i,j)] = 0;
			else 
			{
				double dx = a>0.0001 ? abs(i-((j-b)/a)) : 0.0f; // casos perpendiculares (dist=0)
				double dy = abs(j-a*i-b);

				MT[M_idx(offset,i,j)] = (int)sqrt(dx*dx + dy*dy);
			}
		//	printf(" %d", MT[M_idx(offset,i,j)]);
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

void prep_MV(MVector MV, int traj_id, double proj_ang, double a, double b)
{
	for (int x=0; x<MAT_DIM; x++) 
	{
		double yreta = a*x+b;

		for (int y=0; y<MAT_DIM; y++) 
		{
			double ang = (y<yreta) ? sum_angle(proj_ang, -90.0f) : sum_angle(proj_ang, 90.0f) ;
			float2 vec, nv;
			vec.x = 1.0f;
			vec.y = 1.0f;
			nv = rotate_vector(vec, ang);

			MV[M_idx(traj_id, x, y)].x = nv.x;
			MV[M_idx(traj_id, x, y)].y = nv.y;
		}
	}
}

// config m x n, e.g. 3x7
//in: config
//out: array de arrays (matrizes) de inteiros 
MTraj prep_MT_from_config(int m, int n, MVector MV)
{
	int* root = (int*) malloc(sizeof(int)*m*n*MAT_SIZE);
	host_MV = MV = (MVector) malloc(sizeof(float2) * MAT_SIZE * m * n);

	double ang = 180.0/m;

	int d_traj = fator_separacao_trajs(n);	// distancia entre cada trajetória de uma projeção
	printf("\r\nSeparacao entre trajetorias = %d \r\n",d_traj);
	// Projections
	for (int i=0, offset=0; i<m; i++) 
	{
		// Trajectories
		for (int j=0, b_delta=d_traj; j<n; j++,offset++, b_delta+=d_traj) 
		{
			double a, b, n_b;
			get_eq_traj_director(ang*i, &a, &b);
			n_b = b + b_delta - (int)((double)MAT_DIM/2);
			prep_MT_cells(root, offset, a, n_b);
				
			prep_MV(MV, offset, ang*i, a, n_b);
		}
	} 

	return root;
}

MPart prep_MP(int npart)
{
	MPart MP = (MPart) malloc(sizeof(int)*MAT_SIZE);
	srand(time(NULL));

	for (int i=0; i<MAT_SIZE; i++) MP[i] = 0;	// inicializar com 0 particula/celula
	for (int i=0; i<npart; i++) MP[M_idx(0, rand()%MAT_DIM, rand()%MAT_DIM)]++; 
	
	return MP;
}

MTraj CUDA_PREP_COPY_MT(MTraj MT_src, int m, int n)
{
	MTraj dev_root;
	int size = m*n*MAT_SIZE;
	GPUerrchk(cudaMalloc((void**)&dev_root, sizeof(int)*size));
	GPUerrchk(cudaMemcpy(dev_root, MT_src, sizeof(int)*size, cudaMemcpyHostToDevice));
	return dev_root;
}

MVector CUDA_PREP_COPY_MV(MVector MV_src, int m, int n)
{
	MVector dev_MV;
	int size = m*n*MAT_SIZE;
	GPUerrchk(cudaMalloc((void**)&dev_MV, sizeof(float2)*size));
	GPUerrchk(cudaMemcpy(dev_MV, MV_src, sizeof(float2)*size, cudaMemcpyHostToDevice));
	return dev_MV;
}

MPart CUDA_PREP_COPY_MP(MPart MP_src)
{
	MPart dev_p;
	GPUerrchk(cudaMalloc((void**)&dev_p, sizeof(int)*MAT_SIZE));
	GPUerrchk(cudaMemcpy(dev_p, MP_src, sizeof(int)*MAT_SIZE, cudaMemcpyHostToDevice));
	return dev_p;
}

int* prep_intarray(int length, int val)
{
	int* a = (int*) malloc(sizeof(int)*length);
	for (int i=0;i<length;i++) a[i] = val;
	return a;
}

int* prep_int(int val)
{
	int* a = (int*) malloc(sizeof(int));
	*a = val;
	return a;
}

int* CUDA_PREP_COPY_intarray(int* src, int length)
{
	int* c;
	GPUerrchk(cudaMalloc((void**)&c, sizeof(int)*length));
	GPUerrchk(cudaMemcpy(c, src, sizeof(int)*length, cudaMemcpyHostToDevice));
	return c;
}

int* CUDA_PREP_COPY_int(int* src)
{
	int* c;
	GPUerrchk(cudaMalloc((void**)&c, sizeof(int)));
	GPUerrchk(cudaMemcpy(c, src, sizeof(int), cudaMemcpyHostToDevice));
	return c;
}

int *host_npart, *dev_npart;
int *host_ntraj, *host_nproj, *host_nttltraj, *dev_ntraj, *dev_nproj, *dev_nttltraj;
int *host_np_cur, *host_np_stb, *host_traj_stb, *dev_np_cur, *dev_np_stb, *dev_traj_stb;

void test()
{
	int m, n;
	m = 3;
	n = 7;
	n_part = 64;

	host_npart = prep_intarray(1, n_part);
	host_ntraj = prep_intarray(1, m);
	host_nproj = prep_intarray(1, n);
	host_nttltraj = prep_intarray(1, (*host_ntraj)*(*host_nproj));
	host_np_cur = prep_intarray(*host_nttltraj, 0);
	host_np_stb = prep_intarray(*host_nttltraj, 0);
	host_traj_stb = prep_int(0);
	

	dev_npart =		CUDA_PREP_COPY_intarray(host_npart, 1);
	dev_ntraj =		CUDA_PREP_COPY_intarray(host_ntraj, 1);
	dev_nproj =		CUDA_PREP_COPY_intarray(host_nproj, 1);
	dev_nttltraj =	CUDA_PREP_COPY_intarray(host_nttltraj, 1);
	dev_np_cur =	CUDA_PREP_COPY_intarray(host_np_cur, *host_nttltraj);
	dev_np_stb =	CUDA_PREP_COPY_intarray(host_np_cur, *host_nttltraj);
	dev_traj_stb =	CUDA_PREP_COPY_int(host_traj_stb);
	

	n_traj = m*n;
	host_MTraj = prep_MT_from_config(m,n, host_MV);
	host_MT_Sum = (int*) malloc(sizeof(int)*MAT_SIZE);
	dev_MTraj = CUDA_PREP_COPY_MT(host_MTraj, m,n);
	dev_MV = CUDA_PREP_COPY_MV(host_MV, m, n);

	host_MPart = prep_MP(n_part);
	dev_MPart = CUDA_PREP_COPY_MP(host_MPart);

	


//	APSIRT_main_loop(dev_MTraj, dev_MPart, host_MPart, dev_MV, dev_np_stb, dev_np_cur, dev_traj_stb, dev_nproj, dev_ntraj, dev_npart, *host_nttltraj);


	/*int* root = (int*) malloc(sizeof(int)*MAT_SIZE*2);

	double ang = 120.0f;

	double a, b;
	
	get_eq_traj_director(60.0f, &a, &b);
	prep_MT_cells(root, 0, a, b);

	get_eq_traj_director(ang, &a, &b);
	prep_MT_cells(root, 1, a, b);*/
}



GLuint positionsVBO;
struct cudaGraphicsResource *positionsVBO_CUDA;



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

	



	glPointSize(2.0);
	glBegin(GL_POINTS);

	glColor3f(1.0f,1.0f,1.0f);
	glVertex2d(0.0,0.0);

	int maxval = 0;

	for (int i=0; i<MAT_SIZE; i++) host_MT_Sum[i] = 0;

	for (int k=0; k<n_traj; k++) 
	{
		for (int i=0;i<MAT_DIM; i++) 
		{
			for (int j=0;j<MAT_DIM; j++) 
			{
				int val = host_MTraj[M_idx(k,i,j)];
				//float val_norm = abs(1-(val/(double)MAT_DIM));
				

				int nval = host_MT_Sum[M_idx(0,i,j)] += val;
				maxval = nval>maxval? nval:maxval;

				

			}
		}
	}


	for (int i=0;i<MAT_DIM; i++) 
	{
		for (int j=0;j<MAT_DIM; j++) 
		{
			int val = host_MT_Sum[M_idx(0,i,j)];
			float val_norm = abs(1-(val/(double)maxval));
			glColor3f(val_norm,val_norm,val_norm);
			glVertex2d(to_GL_coord(i), to_GL_coord(j));
		}
	}

	for (int k=0; k<n_traj; k++) 
	{
		for (int i=0;i<MAT_DIM; i++) 
		{
			for (int j=0;j<MAT_DIM; j++) 
			{
				int val = host_MTraj[M_idx(k,i,j)];
				if (val == 0) 
				{
					glColor3f(1.0f,0.0f,0.0f);
					glVertex2d(to_GL_coord(i), to_GL_coord(j));
				}
			}
		}
	}


	// DESENHAR PARTICULAS (SOBREPOE TRAJ)

	for (int i=0;i<MAT_DIM; i++) 
		{
			for (int j=0;j<MAT_DIM; j++) 
			{
				int val = host_MPart[M_idx(0,i,j)];
				if (val > 0) 
				{
					glColor3f(0.0f,1.0f,0.0f);
					glVertex2d(to_GL_coord(i), to_GL_coord(j));

				}
			}
		}

	// DESENHAR CAMPO VET
	if (dbg_showvec)
	for (int i=0;i<MAT_DIM; i++) 
	{
		for (int j=0;j<MAT_DIM; j++) 
		{
			float2 vec = host_MV[M_idx(dbg_trajid, i, j)];
			glColor3f(vec.x, vec.y, 0.0f);
			glVertex2d(to_GL_coord(i), to_GL_coord(j));
		}
	}



	glEnd();


	glFlush();	
	Sleep(50);
}
void update()
{
	/*double h = *host_x;
	GPUerrchk(cudaMemcpy(dev_x, host_x, sizeof(double), cudaMemcpyHostToDevice));
	CUDA_work<<<1,1>>>(dev_x);
	GPUerrchk(cudaMemcpy(host_x, dev_x, sizeof(double), cudaMemcpyDeviceToHost));
	h = *host_x;
	//printf("\r\n %f",*host_x);
	
	cudaDeviceSynchronize();*/
	if (dbg_CUDA_active>0) 
	{
		APSIRT_main_loop(dev_MTraj, dev_MPart, host_MPart, dev_MV, dev_np_stb, dev_np_cur, dev_traj_stb, dev_nproj, dev_ntraj, dev_npart, *host_nttltraj, host_traj_stb);
		dbg_CUDA_active--;
		printf("\r\nCiclo %d\r\n",dbg_CUDA_active);
	}



	GPUerrchk(cudaDeviceSynchronize());
	glutPostRedisplay();
}

void keyboard_handler (unsigned char key, int x, int y)
{
	if (key == 27) exit(0);	//ESC = exit
	else if (key == 'v') dbg_showvec = dbg_showvec==0? 1 : 0;
	else if (key == '+') { dbg_trajid = (dbg_trajid+1) < ((*host_ntraj)*(*host_nproj)) ? dbg_trajid+1 : 0 ; printf("\r\nTraj atual = %d",dbg_trajid); }
	else if (key == 'a') { dbg_CUDA_active+= 50; };
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
