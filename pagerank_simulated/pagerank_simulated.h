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

/***** Struct for process information *****/
typedef struct
{
  int rank;
  int size;
  int local_start;
  int local_end;
} ProcessInfo;

/***** Function declarations *****/
void Read_from_txt_file(char* filename, ProcessInfo *proc_info);
void Random_P_E(ProcessInfo *proc_info);
void Simulated_Distributed_PageRank(ProcessInfo *proc_info);
void Exchange_Ranks_File_Based(ProcessInfo *proc_info);
double Compute_Global_Max_Error_File(double local_max_error, ProcessInfo *proc_info);
double Compute_Global_Sum_File(double local_sum, ProcessInfo *proc_info);
void Cleanup_Communication_Files(ProcessInfo *proc_info);