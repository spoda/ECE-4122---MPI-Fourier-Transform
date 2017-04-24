// Distributed two-dimensional Discrete FFT transform
// Saketh Poda
// ECE8893 Project 1


#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <signal.h>
#include <math.h>
#include <mpi.h>

#include "Complex.h"
#include "InputImage.h"

using namespace std;



void Transform1D(Complex* h, int w, Complex* H);
void Transpose(Complex* ogArray, Complex* tpArray, int h, int w);


void Transform2D(const char* inputFN) 
{ // Do the 2D transform here.
  // 1) Use the InputImage object to read in the Tower.txt file and
  //    find the width/height of the input image.
  // 2) Use MPI to find how many CPUs in total, and which one
  //    this process is
  // 3) Allocate an array of Complex object of sufficient size to
  //    hold the 2d DFT results (size is width * height)
  // 4) Obtain a pointer to the Complex 1d array of input data
  // 5) Do the individual 1D transforms on the rows assigned to your CPU
  // 6) Send the resultant transformed values to the appropriate
  //    other processors for the next phase.
  // 6a) To send and receive columns, you might need a separate
  //     Complex array of the correct size.
  // 7) Receive messages from other processes to collect your columns
  // 8) When all columns received, do the 1D transforms on the columns
  // 9) Send final answers to CPU 0 (unless you are CPU 0)
  //   9a) If you are CPU 0, collect all values from other processors
  //       and print out with SaveImageData().
  InputImage image(inputFN);  // Create the helper object for reading the image
  // Step (1) in the comments is the line above.
  // Your code here, steps 2-9
  int rank, numTasks;
  MPI_Comm_size(MPI_COMM_WORLD, &numTasks);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  int picHeight = image.GetHeight();
  int picWidth = image.GetWidth();
  Complex outputArray[picHeight * picWidth], pre2DArray[picWidth*picHeight], transposedArray[picWidth*picHeight];
  Complex* imageArray = image.GetImageData();
  Transform1D(imageArray, picWidth, outputArray); 
  
  const int rowRank = picWidth/numTasks;
  for (int i = 0; i < numTasks; ++i) {
      MPI_Request pReq;
      int  rc = MPI_Isend(&outputArray[rank*picWidth*rowRank], sizeof(Complex)*(picWidth*rowRank), MPI_CHAR, i, 0, MPI_COMM_WORLD,&pReq);
  }
  //Complex pre2DArray[picWidth*picHeight];
//if (rank == 0) {
  for (int j = 0; j < numTasks; ++j) {
      MPI_Status status1; 
      int rc = MPI_Recv(&pre2DArray[picWidth*rowRank*j], sizeof(Complex)*(picWidth*rowRank), MPI_CHAR,j,0,MPI_COMM_WORLD, &status1);
  }
 // image.SaveImageData("MyAfter1D.txt", pre2DArray, picWidth, picHeight);
//}
  
  image.SaveImageData("MyAfter1D.txt", pre2DArray, picWidth, picHeight);
  //Complex transposedArray[picWidth*picHeight];
  Transpose(pre2DArray, transposedArray, picWidth, picHeight);
  Complex DftArray[picWidth*picHeight];
  Complex final2DArray[picWidth*picHeight];
  Transform1D(transposedArray, picWidth, DftArray);

  MPI_Request pReq3;
  int pc = MPI_Isend(&DftArray[rank*picWidth*rowRank], sizeof(Complex)*rowRank*picWidth, MPI_CHAR, 0, 0, MPI_COMM_WORLD,&pReq3);
  if (rank == 0) {
     for (int k = 0; k < numTasks; ++k) {
         MPI_Status status2;
         int rc = MPI_Recv(&final2DArray[k*rowRank*picWidth], sizeof(Complex)*rowRank*picWidth, MPI_CHAR, k, 0, MPI_COMM_WORLD, &status2);
     }}
  Complex finalArrayTransposed[picWidth*picHeight];
  Transpose(final2DArray, finalArrayTransposed, picWidth, picHeight);
  image.SaveImageData("MyAfter2D.txt", finalArrayTransposed, picWidth, picHeight);
  //}

}

void Transform1D(Complex* h, int w, Complex* H)
{
  // Implement a simple 1-d DFT using the double summation equation
  // given in the assignment handout.  h is the time-domain input
  // data, w is the width (N), and H is the output array.
  //
  int rank, numTasks;
  MPI_Comm_size(MPI_COMM_WORLD, &numTasks);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  const int rowRank = w/numTasks;
  const int start = rowRank * rank;
  const int end = (rowRank *(rank+1)-1);
for (int k = start; k <=end; k++) {

   for (int j = 0; j < w; j++) {
       H[(w*k)+j] = Complex(0,0);
       for (int i = 0; i < w; i++) {
           double angle = (2*M_PI*j*i)/(double) w;
           Complex summation(cos(angle), (-1*sin(angle)));
           H[(k*w)+j] = H[(k*w)+j] + (summation * h[(k*w)+i]);
       }
      //realSummation = ((h[(k*w)+i].real * summation.real) - ((h[(k*w)+i].imag * summation.imag)));
      //H[(w*k)+j].real += realSummation;
      //imagSummation = ((h[(w*k)+i].real * summation.imag) + ((h[(w*k)+i].imag * summation.real)));
      //H[(w*k)+j].imag += (imagSummation);   
}
}
}

void Transpose(Complex* ogArray, Complex* tpArray, int h, int w) {

for (int i = 0; i < w; i++) {
    for (int j = 0; j < h; j++) {
        tpArray[j*w+i] = ogArray[i*w+j];
    }
}
}


int main(int argc, char** argv)
{
  string fn("Tower.txt"); // default file name
  if (argc > 1) fn = string(argv[1]);  // if name specified on cmd line
  InputImage image(fn.c_str()); 
  // MPI initialization here
  int rc;
  rc = MPI_Init(&argc, &argv);
  Transform2D(fn.c_str()); // Perform the transform.
  // Finalize MPI here
  MPI_Finalize();

}  
  
 
  
