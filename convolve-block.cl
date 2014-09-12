
#if defined(cl_khr_fp64)
  #pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)
  #pragma OPENCL EXTENSION cl_amd_fp64 : enable
#endif
__constant double gaussian3[] = {0.319167768453859, 0.361664463092282, 0.319167768453859};


__kernel void unsharp_mask3(__global double *image_in, __global double *image_out, __local double image_block[DIM][DIM],
 unsigned int width, unsigned int height, double threshold, double amount)
{
        double blurred;		/* blurred pixel */
	double diff;		/* difference between orignal and blurred pixel */
	int i = get_global_id(0);
	int j = get_global_id(1);
	int i_local = get_local_id(0);
	int j_local = get_local_id(1);
	image_block[j_local][i_local] = image_in[i+j*width];
	barrier(CLK_LOCAL_MEM_FENCE);
			blurred = gaussian3[0]*image_block[j-1][i] + 
			  gaussian3[1]*image_in[j][i] +
			  gaussian3[2]*image_in[j+1][i] +
			  gaussian3[0]*image_in[j][i-1] +
			  gaussian3[1]*image_in[j][i] +
			  gaussian3[2]*image_in[j][i+1];

			//diff = fabs(image[i+j*width] - fabs(blurred));
			diff = fabs(image_in[i+j*width] - blurred);
			if (fabs(diff/image_in[i+j*width]) > threshold) {
			  image_out[i+j*width] = image_block[j][i] -
				amount*(image_block[j][i] - diff);
			}
			else {
			  image_out[i+j*width] = image_block[j][i];

			}	
}
