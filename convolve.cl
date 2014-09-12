
#if defined(cl_khr_fp64)
  #pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)
  #pragma OPENCL EXTENSION cl_amd_fp64 : enable
#endif
__constant double gaussian3[] = {0.319167768453859, 0.361664463092282, 0.319167768453859};


__kernel void unsharp_mask3(__global double *image_in, __global double *image_out, unsigned int width, unsigned int height, double threshold, double amount)
{
        double blurred;		/* blurred pixel */
	double diff;		/* difference between orignal and blurred pixel */
	int i = get_global_id(0);
	int j = get_global_id(1);
			blurred = gaussian3[0]*image_in[i*width+j-1] + 
			  gaussian3[1]*image_in[i*width+j] +
			  gaussian3[2]*image_in[i*width+j+1] +
			  gaussian3[0]*image_in[(i-1)*width+j] +
			  gaussian3[1]*image_in[i*width+j] +
			  gaussian3[2]*image_in[(i+1)*width+j];

			//diff = fabs(image[i*width+j] - fabs(blurred));
			diff = fabs(image_in[i*width+j] - blurred);
			if (fabs(diff/image_in[i*width+j]) > threshold) {
			  image_out[i*width+j] = image_in[i*width+j] -
				amount*(image_in[i*width+j] - diff);
			}
			else {
			  image_out[i*width+j] = image_in[i*width+j];

			}	
}
