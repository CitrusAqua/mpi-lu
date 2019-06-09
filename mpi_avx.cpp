
#include <iostream>
#include <mpi.h>
#include <x86intrin.h>
#include <sys/time.h>
#include <immintrin.h>

using namespace std;


float** construct_matrix(const int N)
{
    srand((unsigned)time(0));                     
    float** A=new float*[N];
    for(int i=0;i<N;i++)
    {
        A[i]= (float*)_mm_malloc(sizeof(float)*N,16);
        for(int j=0;j<N;j++)
        {
            A[i][j]=(float)rand() / (RAND_MAX>>8);
        }

    }
    return A;

}
float** construct_AVX_matrix(const int N)
{
    srand((unsigned)time(0));                     //���ﲻʹ��������ͻ�������� ����ֳ���1��0֮�������Ϊnan ����ʹ��������ķ���
    float** A=new float*[N];
    for(int i=0;i<N;i++)
    {
        A[i]= (float*)_mm_malloc(sizeof(float)*N,32);
        for(int j=0;j<N;j++)
        {
            A[i][j]=(float)rand() / (RAND_MAX>>8);
        }

    }
    return A;

}
void make_same(float**a,float**b,int length)
{
    for(int i=0;i<length;i++)
    {
        for(int j=0;j<length;j++)
        {
            b[i][j]=a[i][j];
        }
        //cout<<endl;

    }

}
void display(int length,float **b)
{
    for(int i=0;i<length;i++)
    {
        for(int j=0;j<length;j++)
        {
            cout<<b[i][j]<<" ";
        }
        cout<<endl;

    }

}
void Normal_LU(int n,float **A)
{

    for(int k=0;k<n;k++)
    {
        for(int j=k+1;j<n;j++)
        {
            A[k][j]=A[k][j]/A[k][k];
        }
        A[k][k]=1.0;
        for(int i=k+1;i<n;i=i+1)
        {

            for(int j=k+1;j<n;j++)
            {
                A[i][j]=A[i][j]-A[i][k]*A[k][j];
            }
            A[i][k]=0;
        }

    }
}
int N=1024;
void row_node(int rank);
void eliminate(float **block,float **rblock,int r_num1,int r_num2,int r_count,int id);
int main(int argc, char ** argv)
{
    int rank;
	
    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    row_node(rank);
    
    MPI_Finalize();
    return 0;
}
//那个位置创建线程
void row_node(int rank)
{

    if(rank==0)
    {
        int size;  // 线程总数
        MPI_Comm_size(MPI_COMM_WORLD,&size);
        int r_n=(N-N%size)/size;
        struct timeval start_time, end_time;
        //创建矩阵
        float **A=construct_AVX_matrix(N);
        float **B=construct_matrix(N);
        //display(N,A);
        cout<<"********************"<<endl;
        //display(N,B);
        make_same(A,B,N);
        gettimeofday(&start_time,NULL);
        //首先将分配给各个node
        int count=1;
        //cout<<"yes0"<<endl;
        bool flag=false;
        for(int i = r_n; i<N; i=i+1)
        {
            if((count+1)*r_n==i)
            {
                if(flag==false&&count!=size-1)
                {
                    count=count+1;
                }
                //cout<<"i is "<<i<<endl;
            }
            //cout<<count<<endl;
            //cout<<"id:"<<count<<endl;
            if(count==size-1)
            {
                flag=true;//后面count不变都发给最后一个结点
            }
            //然后将每一行发送出去 count为结点id
            MPI_Send(A[i],N,MPI_FLOAT,count,0,MPI_COMM_WORLD);
        }
        //cout<<"yes1"<<endl;
        //然后对第一行进行操作进行除法和消去再发送
        float kk[8] __attribute__((aligned(32)))={0,0,0,0,0,0,0,0};
        __m256* t2=(__m256*)kk;
        for(int k=0; k < r_n; k++)
        {
            // division
            kk[0]=A[k][k];
            kk[1]=A[k][k];
            kk[2]=A[k][k];
            kk[3]=A[k][k];
            kk[4]=A[k][k];
            kk[5]=A[k][k];
            kk[6]=A[k][k];
            kk[7]=A[k][k];
            for(int j=k+1;j<N;j=j+8)
            {

                if(j%8==0)
                {
                    __m256* t1 = (__m256*)&A[k][j];
                    *t1=_mm256_div_ps(*t1,*t2);
                }
                else
                {
                    for(int extra=j;extra<j-(j%8)+8;extra++)
                    {
                        A[k][extra]=A[k][extra]/A[k][k];
                    }
                    j=j-(j%8);
                }
            }
            A[k][k] = 1.0;
            //先做除法 做完除法就发给后面的结点做消去
            for(int id=1;id<size;id++)
            {
                MPI_Send(A[k], N, MPI_FLOAT, id, 0, MPI_COMM_WORLD);

            }
            
            
            //再做消去
            if((r_n-1-k)<=0)
            {
                break; //只有一行不需要消去
            }
            for(int i=k+1;i<r_n;i++)
            {
                kk[0]=A[i][k];
                kk[1]=A[i][k];
                kk[2]=A[i][k];
                kk[3]=A[i][k];
                kk[4]=A[i][k];
                kk[5]=A[i][k];
                kk[6]=A[i][k];
                kk[7]=A[i][k];
                for(int j=k+1;j<N;j=j+8)
                {
                    if(j%8==0)
                    {
                        __m256* tij=(__m256*)&A[i][j];
                        __m256* tkj=(__m256*)&A[k][j];
                        __m256 mul_result;
                        mul_result=_mm256_mul_ps(*t2,*tkj);
                        *tij=_mm256_sub_ps(*tij,mul_result);
                    }
                    else{
                        for(int extra=j;extra<j-(j%8)+8;extra++)
                        {
                            A[i][extra]=A[i][extra]-A[i][k]*A[k][extra];

                        }
                        j=j-(j%8);
                    }
                }
                A[i][k]=0.0;
            }
           
        }
        cout<<"no prob"<<endl;
        //cout<<"yes2"<<endl;
        //然后接收传回来的行
        int count1=1;
        flag=false;
        for(int i=r_n; i<N; i++)
        {
            if((count1+1)*r_n==i)
            {
                if(flag==false&&count1!=size-1)
                {
                    count1=count1+1;
                }
            }
            if(count1==size-1)
            {
                flag=true;
            }                    
            MPI_Recv(A[i], N, MPI_FLOAT, count1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }


        gettimeofday(&end_time,NULL);
        unsigned long time_interval = 1000000*(end_time.tv_sec - start_time.tv_sec) + end_time.tv_usec - start_time.tv_usec;
        cout<<time_interval<<endl;

        Normal_LU(N,B);
        //display(N,A);
        cout<<"*********************************************************";
        //display(N,B);

        
    }
    else
    {
        
        /*
        int size;  // 线程总数
        MPI_Comm_size(MPI_COMM_WORLD,&size);
        int r_n=(N-N%size)/size;
        */
        int size;  // 线程总数
        MPI_Comm_size(MPI_COMM_WORLD,&size);
        //cout<<"szie:"<<size<<endl;
        int r_n=(N-N%size)/size;
        int begin=rank*r_n;
        int end=0;
        if(rank==size-1)
        {
            end=N-1;
        }
        else
        {
            end=begin+r_n-1;
        }
        // 创建空间存放要负责行
        int block_size = end - begin + 1;
        float **block = new float*[block_size];
        for(int i=0; i<block_size; i++)
            block[i] = (float*)_mm_malloc(sizeof(float)*N,32);
        //接收初始值
        //cout<<"yes3"<<endl;
        for(int i=0;i<block_size;i++)
        {
            MPI_Recv(block[i],N, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        }

        //创建空间来接收别的结点发来的行r_row
        //注意这里 每一个块的行号是要减去begin 列号不用 
        float *rrow=(float*)_mm_malloc(sizeof(float)*N,32);
        //接收数据
        float kk[8] __attribute__((aligned(32)))={0,0,0,0,0,0,0,0};
        __m256* t2=(__m256*)kk;
        cout<<"no problem"<<endl;

        for(int k=0; k<begin; k++)  
        {
            //找到第k行是哪一个结点发送
            //cout<<k<<endl;
            int origin = k/r_n;
            //cout<<origin<<endl;
            MPI_Recv(rrow, N, MPI_FLOAT, origin, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            //消去
            //cout<<origin<<endl;
        
            for(int i=0; i<block_size; i++)
            {
                kk[0]=block[i][k];
                kk[1]=block[i][k];
                kk[2]=block[i][k];
                kk[3]=block[i][k];
                kk[4]=block[i][k];
                kk[5]=block[i][k];
                kk[6]=block[i][k];
                kk[7]=block[i][k];
                /*
                for(int j=k+1; j<N; j=j+4)
                    block[i][j] = block[i][j] - block[i][k]*rrow[j];
                */
                
                for(int j=k+1; j<N; j=j+8)
                {
                    if(j%8==0)
                    {
                        
                        __m256* tij=(__m256*)&block[i][j];
                        //cout<<"yes1"<<endl;
                        __m256* tkj=(__m256*)&rrow[j];
                        //cout<<"yes2"<<endl;
                        __m256 mul_result;
                        mul_result=_mm256_mul_ps(*t2,*tkj);
                        //cout<<"yes3"<<endl;
                        *tij=_mm256_sub_ps(*tij,mul_result);
                        //cout<<"yes4"<<endl;
                    }
                    else{
                        for(int extra=j;extra<j-(j%8)+8;extra++)
                        {
                            block[i][extra]=block[i][extra]-block[i][k]*rrow[extra];

                        }
                        j=j-(j%8);
                    }
                }
                block[i][k] = 0.0;
            }

        }
        cout<<"no_problem1"<<endl;
        /*
        if(rank==7)
        {
            cout<<"rank7 yes1"<<endl;
        }*/
        //cout<<"yes5"<<endl;
        //消除完对自己进行行除法和消除然后发出去
        for(int k=begin;k<=end;k++)
        {

           // division
            int i = k - begin;
            kk[0]=block[i][k];
            kk[1]=block[i][k];
            kk[2]=block[i][k];
            kk[3]=block[i][k];
            kk[4]=block[i][k];
            kk[5]=block[i][k];
            kk[6]=block[i][k];
            kk[7]=block[i][k];
            /*
            for(int j=k+1; j<N; j++)
                block[i][j] = block[i][j] / block[i][k];
            block[i][k] = 1.0;
            */
            for(int j=k+1;j<N;j=j+8)
            {

                if(j%8==0)
                {
                    __m256* t1 = (__m256*)&block[k-begin][j];
                    *t1=_mm256_div_ps(*t1,*t2);
                }
                else
                {
                    for(int extra=j;extra<j-(j%8)+8;extra++)
                    {
                        block[k-begin][extra]=block[k-begin][extra]/block[k-begin][k];
                    }
                    j=j-(j%8);
                }
            }
            block[i][k] = 1.0;


            // 跟前面一样进行发出 给后面的结点
            if(rank!=size-1)
            {
                for(int id=rank+1;id<size;id++)
                {
                    MPI_Send(block[i], N, MPI_FLOAT, id, 0, MPI_COMM_WORLD);    
                    
                }
                //然后对自己进行消去
            }
            //  

            for(int true_i=i+1; true_i<block_size; true_i++)
            {
                kk[0]=block[true_i][k];
                kk[1]=block[true_i][k];
                kk[2]=block[true_i][k];
                kk[3]=block[true_i][k];
                kk[4]=block[true_i][k];
                kk[5]=block[true_i][k];
                kk[6]=block[true_i][k];
                kk[7]=block[true_i][k];
                /*
                for(int j=k+1; j<N; j++)
                    block[true_i][j] =block[true_i][j] - block[true_i][k]*block[i][j];
                */
                for(int j=k+1;j<N;j=j+8)
                {
                    if(j%8==0)
                    {
                        __m256* tij=(__m256*)&block[true_i][j];
                        __m256* tkj=(__m256*)&block[i][j];
                        __m256 mul_result;
                        mul_result=_mm256_mul_ps(*t2,*tkj);
                        *tij=_mm256_sub_ps(*tij,mul_result);
                    }
                    else{
                        for(int extra=j;extra<j-(j%8)+8;extra++)
                        {
                            block[i][extra]=block[i][extra]-block[i][k]*block[i][extra];

                        }
                        j=j-(j%8);
                    }
                }
                block[true_i][k] = 0.0;
            }

        }
        //cout<<rank<<" yes6"<<endl;
        for(int i=0; i<block_size; i++)
        {
            MPI_Send(block[i], N, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
        }



    

    }

        



    

}
