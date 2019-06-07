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
//那个位置创建线程
void row_node(int rank)
{
    int size;  // 线程总数
    MPI_Comm_size(MPI_COMM_WORLD,&size);
    //有一部分线程会多出几个
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
    float *rrow=new float[N];


    if(rank==0)
    {
        
        struct timeval start_time, end_time;
        //创建矩阵
        float **A=construct_matrix(N);
        float **B=construct_matrix(N);
        //display(N,A);
        cout<<"********************"<<endl;
        //display(N,B);
        make_same(A,B,N);
        gettimeofday(&start_time,NULL);
        //其实每个节点本质上是一样的除了第一个节点要做一些初始化的工作
        
        //分配 第一个节点要进行分配工作
        int my_i=0;
        for(int i=0;i<N;i++)
        {
            //这个是属于我的行
            int id=i%size;
            if(id==rank)
            {
                //我直接把它保存起来
                // 错误写法 这样要出问题block[my_i]=A[i];
                row_copy(my_i,i,block,A,N);
                my_i++;
            }
            else{
                //不是我的发送出去
                MPI_Send(A[i],N,MPI_FLOAT,id,0,MPI_COMM_WORLD);
            }
        }
        //分配完成后进行接收和消去
        my_i=0;
        for(int k=0;k<N;k++)
        {
            //cout<<"全局k:"<<k<<endl;
            int id=k%size;
            int remaining=N-k; //最后有些不能发
            //cout<<"id:"<<id<<endl;
            if(id==rank) //该我进行除法
            {
                //cout<<"id:"<<id<<endl;
                for(int j=k+1; j<N; j++)
                    block[my_i][j] = block[my_i][j] / block[my_i][k];
                block[my_i][k] = 1.0;
                //然后进行发送
                if(remaining>=size)
                {
                    for(int node=0;node<size;node++)
                    {
                        if(node!=rank)//简单来说是自己就不发
                        {
                            MPI_Send(block[my_i],N,MPI_FLOAT,node,0,MPI_COMM_WORLD);
                        }
                    }
                }
                else
                {
                    for(int node=0;node<remaining;node++)
                    {
                        int dest=(node+rank)%size;
                        if(dest!=rank)//简单来说是自己就不发
                        {
                            MPI_Send(block[my_i],N,MPI_FLOAT,dest,0,MPI_COMM_WORLD);
                        }
                    }

                }
                //cout<<"send_complete"<<endl;
                //对自己进行消去
                for(int i=my_i+1;i<block_size;i=i+1)
                {

                    for(int j=k+1;j<N;j++)
                    {
                        block[i][j]=block[i][j]-block[i][k]*block[my_i][j];
                    }
                    block[i][k]=0;
                }
                //cout<<"eli_complete"<<endl;
                my_i=my_i+1;
                if(my_i==block_size)
                {
                    //已经做完了
                    break;
                }
            }
            else
            {
                //这里是不该我进行除法的情况
                //阻塞接收 接收别人发来的 再操作
                MPI_Recv(rrow,N,MPI_FLOAT,id,0,MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                //收到过后进行消去
                for(int i=my_i;i<block_size;i=i+1)
                {

                    for(int j=k+1;j<N;j++)
                    {
                        block[i][j]=block[i][j]-block[i][k]*rrow[j];
                    }
                    block[i][k]=0;
                }


            }
            //cout<<"my_i:"<<my_i<<endl;

        }
        //做完过后进行最后的接收部分
        int count=0;
        for(int i=0;i<N;i++)
        {
            //cout<<i<<endl;
            int id=i%size;
            if(rank!=id) //数据不在该节点上 在其它节点上面 我们需要从其他节点来接收
            {
                MPI_Recv(A[i],N,MPI_FLOAT, id, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            }
            else // 本地在block_matrix上面
            {               
                for(int j = 0; j<N; j++)
                    A[i][j] =  block[count][j];
                count++;
            }

        }
        cout<<"finish_main"<<endl;
        
       


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

        //先接收数据从 0
        
        for(int i=0;i<block_size;i++)
        {
            MPI_Recv(block[i],N,MPI_FLOAT,0,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        }

        //cout<<rank<<" yes!"<<endl;
        int my_i=0; //看我自己除法做到哪一行了 注意这里是block里面的位置 不是全局位置
        for(int k=0;k<N;k++)
        {
            /*
            if(rank==2)
            {
                cout<<"rank2:"<<k<<endl;
            }*/
            int remaining=N-k; //最后有些不能发
            int id=k%size;

            if(id==rank) //该我进行除法
            {
                for(int j=k+1; j<N; j++)
                    block[my_i][j] = block[my_i][j] / block[my_i][k];
                block[my_i][k] = 1.0;
                //然后进行发送
                if(remaining>=size)
                {
                    for(int node=0;node<size;node++)
                    {
                        
                        if(node!=rank)//简单来说是自己就不发
                        {
                            MPI_Send(block[my_i],N,MPI_FLOAT,node,0,MPI_COMM_WORLD);
                        }
                    }
                }
                else{
                    for(int node=0;node<remaining;node++)
                    {
                        
                        int dest=(node+rank)%size;

                        if(dest!=rank)//简单来说是自己就不发
                        {
                            MPI_Send(block[my_i],N,MPI_FLOAT,dest,0,MPI_COMM_WORLD);
                        }
                    }
                }
                //对自己进行消去
                for(int i=my_i+1;i<block_size;i=i+1)
                {

                    for(int j=k+1;j<N;j++)
                    {
                        block[i][j]=block[i][j]-block[i][k]*block[my_i][j];
                    }
                    block[i][k]=0;
                }
                my_i=my_i+1;
                if(my_i==block_size)
                {
                    //已经做完了
                    break;
                }
            }
            else
            {
                

                //这里是不该我进行除法的情况
                //阻塞接收 接收别人发来的 再操作
                MPI_Recv(rrow,N,MPI_FLOAT,id,0,MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                //收到过后进行消去

                for(int i=my_i;i<block_size;i=i+1)
                {

                    for(int j=k+1;j<N;j++)
                    {
                        block[i][j]=block[i][j]-block[i][k]*rrow[j];
                    }
                    block[i][k]=0;
                }


            }

        }
        //cout<<rank<<"rank"<<"finish"<<endl;
        //最后发送到0结点

        for(int i = 0; i<block_size; i++)
        {
            MPI_Send(block[i], N, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
        }        






        

    

    }

        



    

}
