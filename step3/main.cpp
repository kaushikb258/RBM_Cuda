#include <iostream>
#include <math.h>
#include <string>
#include <stdlib.h>
#include <time.h> 
#include "consts.h"
#include "funcs.h"
#include "adv.h"
#include "out_vtk.h"
#include "matrix.h"

using namespace std;


void matrixMult(double*,double*,double*,int,int,int);


int main()
{
 
//-----
// read offline data from step2

 double *qm = new double[ndof*nmodes]; 
 read_qm(qm); 
 
 double *qtil0 = new double[nmodes];
 read_qtil0(qtil0);

 double *B_deim = new double[nmodes*ndeim]; 
 read_B(B_deim); 

 double *P_deim = new double[ndof*ndeim]; 
 read_P(P_deim); 

 int *pvector = new int[ndeim];
 read_pvector(pvector);

//-----
// compute iloc, jloc, eqnno

 int *iloc = new int[ndeim];
 int *jloc = new int[ndeim];
 int *eqnno = new int[ndeim];
 compute_ijloc(pvector,iloc,jloc,eqnno);
 
//-----
// compute PT

 double *PT = new double[ndeim*ndof];
 computeTranspose(P_deim,PT,ndof,ndeim);
 

// compute qmT

 double *qmT = new double[nmodes*ndof];  
 computeTranspose(qm,qmT,ndof,nmodes);


//-----
// RBM

 // initial qtil
 double *qtil = new double[nmodes];
 for (int i=0; i<nmodes; i++)
 {
  qtil[i] = qtil0[i];
 } 

 // memory for arrays
 double *q = new double[ndof];
 double *Lin = new double[ndof];
 double *VTLin = new double[nmodes];
 double *BPTF = new double[nmodes];
 double *PTF = new double[ndeim];


 clock_t t1, t2; 

 // start RBM
 t1 = clock(); 

 for (int n=0; n<nmax; n++)
 {

  // set arrays to zero
  setToZero(q,ndof);
  setToZero(Lin,ndof);
  setToZero(VTLin,nmodes);
  setToZero(BPTF,nmodes);
  setToZero(PTF,ndeim);
 
  // qm*qtil = q
  matrixMult(qm,qtil,q,ndof,nmodes,1); 

  // ensure mass fractions and T > 0
  makeGTZero(q,ndof);

  // advance one time step
  advance(q,Lin,PTF,pvector,iloc,jloc,eqnno);

  // qmT*Lin = VTLin
  matrixMult(qmT,Lin,VTLin,nmodes,ndof,1);

  // B*PTF = BPTF
  matrixMult(B_deim,PTF,BPTF,nmodes,ndeim,1);

  // update
  update(qtil,VTLin,BPTF);

 }

 t2 = clock(); 

 // qm*qtil = q
 matrixMult(qm,qtil,q,ndof,nmodes,1);
 makeGTZero(q,ndof);

 // output to vtk
 out_vtk(q);
 

 cout<<"Total RBM simulation time (sec) "<<((float)(t2-t1))/CLOCKS_PER_SEC<<"\n";

//-----
// delete memory

 delete [] qm;
 delete [] qtil0;
 delete [] B_deim;
 delete [] P_deim;
 delete [] pvector;
 delete [] iloc;
 delete [] jloc;
 delete [] eqnno;
 delete [] PT;
 delete [] qmT;
 delete [] qtil;
 delete [] q;
 delete [] Lin;
 delete [] VTLin;
 delete [] BPTF;
 delete [] PTF;

}



//----------------------------------------------

void matmul_cpu(int n1, int n2, int n3, double *A, double *B, double *C)
{

 int i, j, k;
 
 for(i=0; i<n1; ++i)
 for(j=0; j<n3; ++j)
 for(k=0; k<n2; ++k)
 {
  C[j+i*n3]+=A[k+i*n2]*B[k*n3+j];
 } 
  
}

//--------------------------------------------

void matrixMult(double *A, double *B, double *C, int n1, int n2, int n3)
{
 
 // set C = 0
 setToZero(C,n1*n3);


#ifdef USEGPU

 // gpu version
 matmul_gpu(n1,n2,n3,A,B,C);   

#else

 // cpu version
 matmul_cpu(n1,n2,n3,A,B,C);    
 
#endif

}

//----------------------------------------------



