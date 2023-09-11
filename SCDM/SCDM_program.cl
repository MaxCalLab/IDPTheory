__kernel void SCDM_ij(
        const int N,
        __global float* SCDM_array,
        __global float* q_list)
{
    int n;
    int m;
    int i = get_global_id(0);
    int j = get_global_id(1);
    double temp_SCDM = 0;

    if ((i < N) && (j < N))
    {
        for (m=j;m<i+1;m++)
        {
                for (n=0;n<j;n++)
                {
                    temp_SCDM += q_list[m] * q_list[n] * pow((float) (m-j), (float) 2.0) /  pow((float) (m-n), (float) 1.5);
                }
        }
        for (m=j+1;m<i+1;m++)
        {
                for (n=j;n<m;n++)
                {
                    temp_SCDM += q_list[m] * q_list[n] * pow((float) (m-n), (float) 0.5);
                }
        }
        for (m=i+1;m<N;m++)
        {
                for (n=0;n<j;n++)
                {
                    temp_SCDM += q_list[m] * q_list[n] * pow((float) (i-j), (float) 2.0) /  pow((float) (m-n), (float) 1.5);
                }
        }
        for (m=i+1;m<N;m++)
        {
                for (n=j;n<i+1;n++)
                {
                    temp_SCDM += q_list[m] * q_list[n] * pow((float) (i-n), (float) 2.0) /  pow((float) (m-n), (float) 1.5);
                }
        }
        SCDM_array[i*N +j] = (float) temp_SCDM;
    }

}