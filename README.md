# Parallel Programming
Basic OpenMP, Pthread and CUDA examples

## Compile
`gpcc`: script to compile OpenMP/Pthread program.
`nvcc`: script to compile CUDA program.

### basic-pp
Contains basic introduction to the threading tools including OpenMP and Pthread
`frogger.c`: the classic frogger game written in C/Pthread
  
### mandelbrot-set
Classic pp problem. A particular set of complex numbers which has a highly convoluted fractal boundary when plotted.
`run_all`: the script to run five versions of the implementation.
  
### CUDA
CUDA introductions and implementation of the n-body problem. n-body problem is the problem of predicting the individual motions of a group of celestial objects interacting with each other gravitationally.

### k-nearest-neighbors
Find k nearest neighbors using Pthread/CUDA
