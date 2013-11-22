#include <cstdlib>
#include <iostream>

int* prep_intarray(int length, int val)
{
	int* a = (int*) malloc(sizeof(int)*length);
	for (int i=0;i<length;i++) a[i] = val;
	return a;
}

int* read_sinogram(char* path, int *nproj, int *ntraj, int* npart)
{
	int* np_stb;
	FILE * pFile;
	
	pFile = fopen (path,"r");
	if (pFile != NULL) 
	{		
		fscanf (pFile, "%d", nproj);
		fscanf (pFile, "%d", ntraj);

		np_stb = prep_intarray((*nproj)*(*ntraj), 0);

		for (int i=0; i<*nproj*(*ntraj); i++) fscanf (pFile, "%d", &(np_stb[i]) );
			
		fscanf (pFile, "%d", npart);
		fclose (pFile);
	}
	else
	{
		printf("\r\nERROR: FILE NOT FOUND");
	}

	return np_stb;
}
