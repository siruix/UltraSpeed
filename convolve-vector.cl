
#if defined(cl_khr_fp64)
  #pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)
  #pragma OPENCL EXTENSION cl_amd_fp64 : enable
#endif
__constant double gaussian3[] = {0.319167768453859, 0.361664463092282, 0.319167768453859};


__kernel void unsharp_mask3(__global double4 *image_in, __global double4 *image_out, unsigned int width, unsigned int height, double4 threshold, double4 amount)
{
    double4 blurred;		/* blurred pixel */
	double4 diff;		/* difference between orignal and blurred pixel */
	int i = get_global_id(0)*4;
	int j = get_global_id(1);
			blurred = gaussian3[0]*image_in[i+(j-1)*width] + 
			  gaussian3[1]*image_in[i+j*width] +
			  gaussian3[2]*image_in[i+(j+1)*width] +
			  gaussian3[0]*image_in[i-1+j*width] +
			  gaussian3[1]*image_in[i+j*width] +
			  gaussian3[2]*image_in[i+1+j*width];

			//diff = fabs(image[i+j*width] - fabs(blurred));
			diff = fabs(image_in[i+j*width] - blurred);
			if (fabs(diff/image_in[i+j*width]) > threshold) {
			  image_out[i+j*width] = image_in[i+j*width] -
				amount*(image_in[i+j*width] - diff);
			}
			else {
			  image_out[i+j*width] = image_in[i+j*width];

			}	
}
