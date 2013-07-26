#include "PSIRT.cu"

inline void GPUassert(cudaError_t code, char * file, int line, bool Abort=true)
{
    if (code != 0) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code),file,line);
        if (Abort) exit(code);
    }       
}


__device__ PSIRT* build_psirt_from_parameters(PSIRT* dev_psirt, Trajectory* t, Particle* p, int n_proj, int n_traj, int n_part);


__device__ PSIRT* build_psirt_from_parameters(PSIRT* dev_psirt, Trajectory* t, Particle* p, int n_proj, int n_traj, int n_part)
{
	
}
