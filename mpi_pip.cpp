#include <iostream>
#include <mpi.h>
#include <x86intrin.h>
#include <sys/time.h>

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
void row_copy(int row1,int row2,float **m1,float **m2,int N)
{
    for(int j=0;j<N;j++)
    {
        m1[row1][j]=m2[row2][j]; //复制一下
    }
}
bool validation(float **A,float **B,int N)
{
    for(int i=0;i<N;i++)
    {
        for(int j=0;j<N;j++)
        {
            if(A[i][j]!=B[i][j])
            {
                return false;
            }
        }
    }
    return true;
}
int N=1000;
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
void row_node(int rank)
{
    int size;  // 线程总数
    MPI_Comm_size(MPI_COMM_WORLD,&size);
    //有一部分线程会多出几个
    /*
    int extra=N%size;
    int r_n=(N-extra)/size;
    int block_size;
    if(rank<extra)
    {
        block_size=r_n+1;//小于余数加1

    }
    else
    {
        block_size=r_n;

    }
    //注意这里每一个块只保存自己块占有的那一部分
    float **block=new float*[block_size];
    for(int i=0;i<block_size;i++)
    {
        block[i]=new float[N];
    }
    float *rrow=new float[N];*/


    if(rank==0)
    {
        int size;  // 线程总数
        MPI_Comm_size(MPI_COMM_WORLD,&size);
        int r_n=(N-N%size)/size;
        struct timeval start_time, end_time;
        //创建矩阵
        float **A=construct_matrix(N);
        float **B=construct_matrix(N);
        //display(N,A);
        cout<<"********************"<<endl;
        //display(N,B);
        make_same(A,B,N);
        gettimeofday(&start_time,NULL);
        
        //
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
        // 先做除法操作，然后发送结果 ，再消去自己的。
        for(int k=0; k < r_n; k++)
        {
            // division
            for(int j=k+1; j<N; j++)
                A[k][j] = A[k][j] / A[k][k];
            A[k][k] = 1.0;


            // 将division结果发给下一个节点
            int dest = rank + 1;
            MPI_Send(A[k], N, MPI_FLOAT, dest, 0, MPI_COMM_WORLD);
            
            if((r_n-1-k)<=0)
            {
                break; //只有一行不需要消去
            }
            for(int i=k+1;i<r_n;i=i+1)
            {

                for(int j=k+1;j<N;j++)
                {
                    A[i][j]=A[i][j]-A[i][k]*A[k][j];
                }
                A[i][k]=0;
            }
            
        }
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
        bool a=validation(A,B,N);
        if(a==true)
        {
            cout<<"yes"<<endl;
        }

        
    }
    else
    {
        int size;  // 线程总数
        MPI_Comm_size(MPI_COMM_WORLD,&size);
        int r_n=(N-N%size)/size;  
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
            block[i] = new float[N];
        //接收初始值
        //cout<<"yes3"<<endl;
        for(int i=0;i<block_size;i++)
        {
            MPI_Recv(block[i],N, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE); 
        }    
        float *rrow=new float[N];
        //接收数据
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
                for(int j=k+1; j<N; j++)
                    block[i][j] = block[i][j] - block[i][k]*rrow[j];
                block[i][k] = 0.0;
            }

        }

        //消除完对自己进行行除法和消除然后发出去
        for(int k=begin;k<=end;k++)
        {
                // division
            int i = k - begin;
            for(int j=k+1; j<N; j++)
                block[i][j] = block[i][j] / block[i][k];
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
                for(int j=k+1; j<N; j++)
                    block[true_i][j] =block[true_i][j] - block[true_i][k]*block[i][j];
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
     

    }

        



    

}
