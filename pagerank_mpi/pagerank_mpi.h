/******************** Structs ********************/

/***** Struct for timestamps *****/
struct timeval start,end;

/***** Struct used for Nodes data *****/

typedef struct
{
  double p_t0;
  double p_t1;
  double e;
  int *To_id;
  int con_size;
}Node;

/***** Struct for communication buffers *****/
typedef struct
{
  int *node_ids;
  double *rank_values;
  int count;
} CommBuffer;

/***** Function declarations *****/
void Read_from_txt_file(char* filename, int rank, int size);
void Random_P_E(int local_start, int local_end);
void Distributed_PageRank(int rank, int size, int local_start, int local_end);
void Exchange_Ranks_Nonblocking(int rank, int size, int local_start, int local_end);
double Compute_Global_Max_Error(double local_max_error, int rank, int size);
double Compute_Global_Sum(double local_sum, int rank, int size);