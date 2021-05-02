#include "driver.hpp"
#include "utils.hpp"
#include <fstream>
#include <iostream>
#include <sstream>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>
#include "ssw.h"
#include <atomic>

size_t gpu_bsw_driver::get_tot_gpu_mem(int id) {
  cudaDeviceProp prop;
  cudaErrchk(cudaGetDeviceProperties(&prop, id));
  return prop.totalGlobalMem;
}



/* This table is used to transform nucleotide letters into numbers. */
int8_t nt_table[128] = {
  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,
  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,
  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,
  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,
  4, 0, 4, 1,  4, 4, 4, 2,  4, 4, 4, 4,  4, 4, 4, 4,
  4, 4, 4, 4,  3, 3, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,
  4, 0, 4, 1,  4, 4, 4, 2,  4, 4, 4, 4,  4, 4, 4, 4,
  4, 4, 4, 4,  3, 3, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4
};

//there really is a lot of crazy coming into this function
// 1 read, 1 contig to be aligned
// the resulting alignment block to point to
// where in that block to point to (see note in function)
// the scoring matrix for ssw, the size of that matrix for ssw (maybe pull it out into static space)
// the start gap scoring
// the end gap scoring

//we are going to pass in our own storage for the numeric versions of the sequences, 2 mallocs a function call is too much.

void cpu_do_one_alignment(std::string read, std::string contig, gpu_bsw_driver::alignment_results *alignments, int alignment_index, const int8_t* mat, int32_t n, short startGap, short extendGap, int8_t* current_read_numeric, int8_t* current_contig_numeric)
{
      int32_t flag=0,filter=0;
      int8_t* table = nt_table;
      const int32_t current_read_length = read.length();
      const int32_t current_contig_length = contig.length();

      // int8_t* current_read_numeric = (int8_t*)malloc(s1);   //this is a sore point in the code, we really want something better... i would be happier with each caller to pass in its own storage.
      // int8_t* current_contig_numeric = (int8_t*)malloc(s1); //taking values from the example, ssw usually does a realloc schme thats weird.

      //just convert the data to the format ssw wants...
      for(int i=0; i < current_read_length; ++i)
      {
        current_read_numeric[i] = table[(int)read[i]];
      }
      for(int i=0; i < current_contig_length; ++i)
      {
        current_contig_numeric[i] = table[(int)contig[i]];
      }

      //create a ssw profile from init
		  s_profile* p; //, *p_rc = 0; //we don't need the reverse compliment i assume.
      s_align* result;
      int32_t maskLen = current_read_length/2; //following the example, this is the mask length, used for suboptimal alignments ??

      //s_profile* ssw_init (const int8_t* read, const int32_t readLen, const int8_t* mat, const int32_t n, const int8_t score_size);
      // score_size = 2 means i dont know,,, 0 < 255, 1 >= 255, both are meaningless to me.
      p = ssw_init(current_read_numeric,current_read_length,mat,n,2);
			result = ssw_align (p, current_contig_numeric, current_contig_length, startGap * -1, extendGap * -1, flag, filter, 0, maskLen);

      //assign the results to the input array, no one shares an alignment index so it should be thread safe....
      //the way the result data structure points to more arrays of individual results makes it difficult to 
      //have this function just return a single result, and let the caller handle it, this function should be
      //more attached to the alignment results data, in either event passint it in is super weird, but this is the explanation.
      alignments->query_begin[alignment_index] = result->ref_begin1;
      alignments->query_end[alignment_index] = result->ref_end1;
      alignments->ref_begin[alignment_index] = result->read_begin1;
      alignments->ref_end[alignment_index] = result->read_end1;
      alignments->top_scores[alignment_index] = result->score1;

      //destroy the result since ssw_init and ssw_align allocates something..
      align_destroy(result);
      init_destroy(p);
}

//strA and strB are cuda allocated storage for all the strings in the batch..., we don't want to re-malloc them for every batch
//offsetA_h and offsetB_h are also cuda mallocs on the host to store i think results...

// NOTE: this might not quite work as a function because we need to do some stealing and this hides the queue from the process...
//       it can be here for now until i get the single pull from the cpu working well then we can maybe shimmy this in well...

void gpu_do_batch_alignments(std::vector<std::string> sequencesA, std::vector<std::string> sequencesB, short scores[4], int batch_size, gpu_bsw_driver::alignment_results *alignments, int alignment_index, char*strA, char*strB,char *strA_d, char *strB_d,unsigned* offsetA_h, unsigned* offsetB_h, unsigned int maxContigSize, unsigned int maxReadSize, cudaStream_t* streams_cuda)
{
    auto packing_start = NOW;

    short matchScore = scores[0], misMatchScore = scores[1], startGap = scores[2], extendGap = scores[3];

    //pointers to where in the alignment results we are going to return into...
    short* alAbeg = alignments->ref_begin + alignment_index;
    short* alBbeg = alignments->query_begin + alignment_index;
    short* alAend = alignments->ref_end + alignment_index;
    short* alBend = alignments->query_end + alignment_index;  // memory on CPU for copying the results
    short* top_scores_cpu = alignments->top_scores + alignment_index;

    gpu_alignments gpu_data(batch_size); //i guess this cleans itself up when it leaves scope...

    int blocksLaunched = batch_size; //0;
    // std::vector<std::string>::const_iterator beginAVec;
    // std::vector<std::string>::const_iterator endAVec;
    // std::vector<std::string>::const_iterator beginBVec;
    // std::vector<std::string>::const_iterator endBVec;

    // std::vector<std::string> sequencesA(beginAVec, endAVec);
    // std::vector<std::string> sequencesB(beginBVec, endBVec);  //we can just pass in seqA and B if this is where things live at the end of the day

    long unsigned running_sum = 0;
    int sequences_per_stream = (blocksLaunched) / NSTREAMS;
    int sequences_stream_leftover = (blocksLaunched) % NSTREAMS;
    long unsigned half_length_A = 0;
    long unsigned half_length_B = 0;

    auto start_cpu = NOW;

    for(int i = 0; i < sequencesA.size(); i++)
    {
        running_sum +=sequencesA[i].size();
        offsetA_h[i] = running_sum;//sequencesA[i].size();
        if(i == sequences_per_stream - 1){
            half_length_A = running_sum;
            running_sum = 0;
          }
    }
    long unsigned totalLengthA = half_length_A + offsetA_h[sequencesA.size() - 1];

    running_sum = 0;
    for(int i = 0; i < sequencesB.size(); i++)
    {
        running_sum +=sequencesB[i].size();
        offsetB_h[i] = running_sum; //sequencesB[i].size();
        if(i == sequences_per_stream - 1){
          half_length_B = running_sum;
          running_sum = 0;
        }
    }
    long unsigned totalLengthB = half_length_B + offsetB_h[sequencesB.size() - 1];

    auto end_cpu = NOW;
    std::chrono::duration<double> cpu_dur = end_cpu - start_cpu;

    //total_time_cpu += cpu_dur.count();
    long unsigned offsetSumA = 0;
    long unsigned offsetSumB = 0;

    //BW NOTE: strA and strB live in host memory, per thread, this is copying the sequences to there.
    //         sequencesA and sequencesB was constructed at the beginning of the loop, it is all the 
    //         query/ref sequences per iteration...

    for(int i = 0; i < sequencesA.size(); i++)
    {
        char* seqptrA = strA + offsetSumA;
        memcpy(seqptrA, sequencesA[i].c_str(), sequencesA[i].size());
        char* seqptrB = strB + offsetSumB;
        memcpy(seqptrB, sequencesB[i].c_str(), sequencesB[i].size());
        offsetSumA += sequencesA[i].size();
        offsetSumB += sequencesB[i].size();
    }

    auto packing_end = NOW;
    std::chrono::duration<double> packing_dur = packing_end - packing_start;

    //total_packing += packing_dur.count();

    asynch_mem_copies_htd(&gpu_data, offsetA_h, offsetB_h, strA, strA_d, strB, strB_d, half_length_A, half_length_B, totalLengthA, totalLengthB, sequences_per_stream, sequences_stream_leftover, streams_cuda);


    //BW NOTE: minSize is the lesser of the biggest query or reference string.
    unsigned minSize = (maxReadSize < maxContigSize) ? maxReadSize : maxContigSize;
    unsigned totShmem = 3 * (minSize + 1) * sizeof(short);
    unsigned alignmentPad = 4 + (4 - totShmem % 4);
    size_t   ShmemBytes = totShmem + alignmentPad;
    if(ShmemBytes > 48000)
        cudaFuncSetAttribute(gpu_bsw::sequence_dna_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, ShmemBytes);

    gpu_bsw::sequence_dna_kernel<<<sequences_per_stream, minSize, ShmemBytes, streams_cuda[0]>>>(
        strA_d, strB_d, gpu_data.offset_ref_gpu, gpu_data.offset_query_gpu, gpu_data.ref_start_gpu,
        gpu_data.ref_end_gpu, gpu_data.query_start_gpu, gpu_data.query_end_gpu, gpu_data.scores_gpu, matchScore, misMatchScore, startGap, extendGap);
    cudaErrchk(cudaGetLastError());

    gpu_bsw::sequence_dna_kernel<<<sequences_per_stream + sequences_stream_leftover, minSize, ShmemBytes, streams_cuda[1]>>>(
        strA_d + half_length_A, strB_d + half_length_B, gpu_data.offset_ref_gpu + sequences_per_stream, gpu_data.offset_query_gpu + sequences_per_stream,
          gpu_data.ref_start_gpu + sequences_per_stream, gpu_data.ref_end_gpu + sequences_per_stream, gpu_data.query_start_gpu + sequences_per_stream, gpu_data.query_end_gpu + sequences_per_stream,
          gpu_data.scores_gpu + sequences_per_stream, matchScore, misMatchScore, startGap, extendGap);
    cudaErrchk(cudaGetLastError());

    // copyin back end index so that we can find new min
    asynch_mem_copies_dth_mid(&gpu_data, alAend, alBend, sequences_per_stream, sequences_stream_leftover, streams_cuda);

    //this also saves out the results of the end locations... into alAend and alBend

    cudaStreamSynchronize (streams_cuda[0]);
    cudaStreamSynchronize (streams_cuda[1]);

    auto sec_cpu_start = NOW;
    int newMin = get_new_min_length(alAend, alBend, blocksLaunched); // find the new largest of smaller lengths
    auto sec_cpu_end = NOW;
    std::chrono::duration<double> dur_sec_cpu = sec_cpu_end - sec_cpu_start;
    //total_time_cpu += dur_sec_cpu.count();

    gpu_bsw::sequence_dna_reverse<<<sequences_per_stream, newMin, ShmemBytes, streams_cuda[0]>>>(
            strA_d, strB_d, gpu_data.offset_ref_gpu, gpu_data.offset_query_gpu, gpu_data.ref_start_gpu,
            gpu_data.ref_end_gpu, gpu_data.query_start_gpu, gpu_data.query_end_gpu, gpu_data.scores_gpu, matchScore, misMatchScore, startGap, extendGap);
    cudaErrchk(cudaGetLastError());

    gpu_bsw::sequence_dna_reverse<<<sequences_per_stream + sequences_stream_leftover, newMin, ShmemBytes, streams_cuda[1]>>>(
            strA_d + half_length_A, strB_d + half_length_B, gpu_data.offset_ref_gpu + sequences_per_stream, gpu_data.offset_query_gpu + sequences_per_stream ,
            gpu_data.ref_start_gpu + sequences_per_stream, gpu_data.ref_end_gpu + sequences_per_stream, gpu_data.query_start_gpu + sequences_per_stream, gpu_data.query_end_gpu + sequences_per_stream,
            gpu_data.scores_gpu + sequences_per_stream, matchScore, misMatchScore, startGap, extendGap);
    cudaErrchk(cudaGetLastError());

    //this copies the results from gpu_data out to a pointer of where the A and B results have their "start" query but also named beg here.
    asynch_mem_copies_dth(&gpu_data, alAbeg, alBbeg, top_scores_cpu, sequences_per_stream, sequences_stream_leftover, streams_cuda);

}


//struct vs std::pair? 
// typedef struct alignment_batch_t{
//     int start;
//     int stop;
// } alignment_batch_t;

// typedef struct AtomicAlignmentQueue{
//   std::atomic_uint64_t current_index;
//   int max_index;
//   AtomicAlignmentQueue() : current_index(0){}
//   alignment_batch_t GetBatch(int count){ alignment_batch_t retVal; retVal.start = current_index.fetch_add(count); retVal.stop = retVal.start + count;};
// } AtomicAlignmentQueue;


//note: OMP Atomic protects a single variable OMP Critical can protect an entire block...

void
gpu_bsw_driver::gpu_cpu_driver_dna(std::vector<std::string> reads, std::vector<std::string> contigs, gpu_bsw_driver::alignment_results *alignments, short scores[4], float factor)
{

    //initialize some values from the original implementation.
    int32_t l,m,k,n=5,s1;
    short matchScore = scores[0], misMatchScore = scores[1], startGap = scores[2], extendGap = scores[3];
    unsigned maxContigSize = getMaxLength(contigs);
    unsigned maxReadSize = getMaxLength(reads);
    
    // "s1" is the variable ssw uses to allocate a buffer for the numeric versions of the sequence.
    // we want to re-use this so we scope our allocation outside the function but per thread.
    // the demo uses the size of int as space, but we know our max sizes immediately so we will do a tighter allocation.
    s1=maxContigSize>maxReadSize?maxContigSize:maxReadSize; 
    
    unsigned totalAlignments = contigs.size(); // assuming that read and contig vectors are same length 
    // not sure actual intention, if we just want the max(reads,contigs) that is not hard but... all tests we have are of same length
    auto read_sequence_ptr = reads.begin();
    auto contig_sequence_ptr = contigs.begin();
    initialize_alignments(alignments, totalAlignments); // pinned memory allocation

    //bw: ssw uses a 5x5 scoring matrix ?!?! maybe for "ambiguous base?"
    //  but the header uses a 4x4 scoring matrix in the example. we will use the demo version.
    int8_t* mata = (int8_t*)calloc(25, sizeof(int8_t));
    const int8_t* mat = mata;

    // initialize scoring matrix for genome sequences --- more copy pasta, 
    for (l = k = 0; (l < 4); ++l) {
      for (m = 0; (m < 4); ++m) mata[k++] = l == m ? matchScore : misMatchScore;
      mata[k++] = 0; // ambiguous base
    }

    int batch_size = 100; //different for gpu/cpu?

    auto start = NOW;

    std::cout<< "SSW GPU-CPU DRIVER STARTED w/" << omp_get_max_threads() << " threads!" << std::endl;
  
    //shared variables, should also only be touched in atomic or critical regions...
    uint64_t work_stolen_count=0;
    uint64_t total_work_alignment_index=0;

    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    std::cout << "Number of GPU Threads: " << deviceCount << std::endl; 

    // std::cout << "Setting up Streams\n"; //hmm maybe these streams need to be global
    // cudaStream_t streams_cuda[NSTREAMS];
    // for(int stm = 0; stm < NSTREAMS; stm++){
    //   cudaStreamCreate(&streams_cuda[stm]);
    // }


    size_t tot_mem_req_per_aln = maxReadSize + maxContigSize + 2 * sizeof(int) + 5 * sizeof(short);

    //creates a parallel region, explicitly stating the variables we want to be shared.
    #pragma omp parallel firstprivate(batch_size) shared(work_stolen_count,total_work_alignment_index)
    {



      //assume one thread per device and those threads share the id with the device.
      int my_cpu_id = omp_get_thread_num();  //we really need to decide on some sort of formating, camel case vs _, choose 1!

      bool hasGPU = my_cpu_id < deviceCount;
      if(hasGPU)
      {
        cudaSetDevice(my_cpu_id);
        int myGPUid;
        cudaGetDevice(&myGPUid);
                
        //float total_time_cpu = 0;

        size_t gpu_mem_avail = get_tot_gpu_mem(myGPUid);
        unsigned max_alns_gpu = floor(((double)gpu_mem_avail*factor)/tot_mem_req_per_aln);
        unsigned max_alns_sugg = 20000;
        max_alns_gpu = max_alns_gpu > max_alns_sugg ? max_alns_sugg : max_alns_gpu;
        batch_size = max_alns_gpu; 
        std::cout<<"Mem (bytes) avail on device "<<myGPUid<<":"<<(long unsigned)gpu_mem_avail<<"\n";
        std::cout<<"Mem (bytes) using on device "<<myGPUid<<":"<<(long unsigned)gpu_mem_avail*factor<<"\n";

        std::cout << "GPU Thread... w/ batch size = " << batch_size << "\n";

        std::cout << "Setting up Streams\n"; //Instances for each GPU
        cudaStream_t streams_cuda[NSTREAMS];
        for(int stm = 0; stm < NSTREAMS; stm++){
          cudaStreamCreate(&streams_cuda[stm]);
        }

        //SETUP CUDA MEMORY  //FIXME?: the mallocs are using a size of a differen't type...
        std::cout << "Setting up CUDA Memory!\n";
        unsigned* offsetA_h;
        cudaMallocHost(&offsetA_h, sizeof(int)* batch_size);
        unsigned* offsetB_h;
        cudaMallocHost(&offsetB_h, sizeof(int)* batch_size);

        char *strA_d, *strB_d;
        cudaErrchk(cudaMalloc(&strA_d, maxContigSize * batch_size * sizeof(char)));
        cudaErrchk(cudaMalloc(&strB_d, maxReadSize * batch_size * sizeof(char)));

        char* strA;
        cudaMallocHost(&strA, sizeof(char)*maxContigSize * batch_size );
        char* strB;
        cudaMallocHost(&strB, sizeof(char)* maxReadSize * batch_size);

        //END CUDA MEMORY

        //END GPU SETUP
        uint64_t atomic_alignment_index;
        #pragma omp atomic read
        atomic_alignment_index = total_work_alignment_index;

        //GPU START WORK
        while(atomic_alignment_index < totalAlignments)
        {

          //********* GPU THREAD WORK
          
          int thread_current_alignment_index_start;
          int thread_current_alignment_index_end;
          //eat up the alignments until the queue is completed.
          //#pragma omp atomic capture
          #pragma omp critical //maybe doing too much for atomic, just use critical for now.
          { 
            thread_current_alignment_index_start=total_work_alignment_index; 
            if(thread_current_alignment_index_start + batch_size < totalAlignments)
            {
              thread_current_alignment_index_end = total_work_alignment_index+=batch_size;
            }
            else
            {
              //take the last "batch"
              total_work_alignment_index = totalAlignments;
              thread_current_alignment_index_end = totalAlignments;

              if(thread_current_alignment_index_start >= totalAlignments)
              {
                //something bad happened, abort. that while loop might not be thread safe?
                std::cout << "Warning Atomic Queue is Broken!" << std::endl;
                //assert
              }
            }
            
          }
          //DO BATCH OF GPU WORK... i also dont think you need to add off the begin, this is a little wild, just index into the vector... actually i think you can just get a range straight up..
          // just do what works and clean up later.
          std::vector<std::string> sequencesA(contigs.begin()+thread_current_alignment_index_start, contigs.begin()+thread_current_alignment_index_end);
          std::vector<std::string> sequencesB(reads.begin()+thread_current_alignment_index_start, reads.begin()+thread_current_alignment_index_end);

           //gpu_do_batch_alignments(sequencesA, sequencesB, scores, batch_size, alignments, thread_current_alignment_index_start, strA, strB, strA_d, strB_d, offsetA_h, offsetB_h, maxContigSize, maxReadSize, streams_cuda);

          #pragma omp atomic read
          atomic_alignment_index = total_work_alignment_index;
        }

        //GPU END WORK

      }
      else
      {
        //if cpu allocate some working memory
        int8_t* current_read_numeric = (int8_t*)malloc(s1);   //this is a sore point in the code, we really want something better... i would be happier with each caller to pass in its own storage.
        int8_t* current_contig_numeric = (int8_t*)malloc(s1); //taking values from the example, ssw usually does a realloc schme thats weird.

      //end cpu setup


    
      uint64_t atomic_alignment_index;
      #pragma omp atomic read
      atomic_alignment_index = total_work_alignment_index;

      while(atomic_alignment_index < totalAlignments)
      {

        //********* CPU THREAD WORK
        
        int thread_current_alignment_index_start;
        int thread_current_alignment_index_end;
        //eat up the alignments until the queue is completed.
        //#pragma omp atomic capture
        #pragma omp critical //maybe doing too much for atomic, just use critical for now.
        { 
          thread_current_alignment_index_start=total_work_alignment_index; 
          if(thread_current_alignment_index_start + batch_size < totalAlignments)
          {
            thread_current_alignment_index_end = total_work_alignment_index+=batch_size;
          }
          else
          {
            //take the last "batch"
            total_work_alignment_index = totalAlignments;
            thread_current_alignment_index_end = totalAlignments;

            if(thread_current_alignment_index_start >= totalAlignments)
            {
              //something bad happened, abort. that while loop might not be thread safe?
              std::cout << "Warning Atomic Queue is Broken!" << std::endl;
              //assert
            }
          }
          
        }

        for(int i=thread_current_alignment_index_start; i < thread_current_alignment_index_end; ++i)
        {

          auto  current_read = *(read_sequence_ptr+i);
          auto  current_contig = *(contig_sequence_ptr+i);

          cpu_do_one_alignment(current_read,current_contig,alignments,i,mat,n,startGap,extendGap,current_read_numeric,current_contig_numeric);
        }

         #pragma omp atomic
         work_stolen_count+= (thread_current_alignment_index_end-thread_current_alignment_index_start);
        //*********


        //TODO DO GPU THREAD WORK, this will pull a batch from the queue but also do the whole cpu thread work routine, between kernel calls
        // to start lets not do that, it might not actually be worth it anyway... just assume rank=0 owns a gpu and only does gpu work.


        #pragma omp atomic read
        atomic_alignment_index = total_work_alignment_index;
      }
      //if cpu free up the memory we used for processing
      free(current_read_numeric);
      free(current_contig_numeric);
    }
        #pragma omp barrier
    }      //end of parallel region.

    free(mata);

    auto end  = NOW;
    std::chrono::duration<double> diff = end - start;
    std::cout << "Total Alignments:"<<totalAlignments<<"\n"<<"Max Reference Size:"<<maxContigSize<<"\n"<<"Max Query Size:"<<maxReadSize<<"\n" <<"Total Execution Time (seconds):"<< diff.count() <<std::endl;
    std::cout << "Work Stolen: " << work_stolen_count << "=> " << round((float)work_stolen_count/(float)totalAlignments * 100) << "%" << std::endl;
}



void
gpu_bsw_driver::cpu_driver_dna(std::vector<std::string> reads, std::vector<std::string> contigs, gpu_bsw_driver::alignment_results *alignments, short scores[4], float factor)
{

    std::cout<< "SSW CPU DRIVER STARTED w/" << omp_get_max_threads() << " threads!" << std::endl;



  	int8_t* table = nt_table;

    //the SSW takes all positive Integers for the scores above, so adjust accordingly.
    int32_t l,m,k,s1=67108864,n=5,filter=0,flag=0;
    short matchScore = scores[0], misMatchScore = scores[1], startGap = scores[2], extendGap = scores[3];
    unsigned maxContigSize = getMaxLength(contigs);
    unsigned maxReadSize = getMaxLength(reads);
    unsigned totalAlignments = contigs.size(); // assuming that read and contig vectors are same length

    initialize_alignments(alignments, totalAlignments); // pinned memory allocation

    //bw: ssw uses a 5x5 scoring matrix ?!?! maybe for "ambiguous base?"
    //  but the header uses a 4x4 scoring matrix in the example. we will use the demo version.
    int8_t* mata = (int8_t*)calloc(25, sizeof(int8_t));
    const int8_t* mat = mata;

    // initialize scoring matrix for genome sequences --- more copy pasta, 
    for (l = k = 0; (l < 4); ++l) {
      for (m = 0; (m < 4); ++m) mata[k++] = l == m ? matchScore : misMatchScore;	/* weight_match : -weight_mismatch */
      mata[k++] = 0; // ambiguous base
    }
    for (m = 0; (m < 5); ++m) mata[k++] = 0;

    //by default we probably have a scoring matrix like below..
    /*
       1 -3 -3 -3  0
      -3  1 -3 -3  0
      -3 -3  1 -3  0
      -3 -3 -3  1  0
       0  0  0  0  0
    */
    auto start = NOW;

   #pragma omp parallel 
   {
   //these are arrays of the numeric version of the sequences
   int8_t* current_read_numeric = (int8_t*)malloc(s1);
   int8_t* current_contig_numeric = (int8_t*)malloc(s1); //taking values from the example, ssw usually does a realloc schme thats weird.

    auto read_sequence_ptr = reads.begin();
    auto contig_sequence_ptr = contigs.begin();

    //while (read_sequence_ptr != reads.end() && contig_sequence_ptr != contigs.end())
    //{
    //auto  current_read = *read_sequence_ptr++;
    //auto  current_contig = *contig_sequence_ptr++;
    int alignment_index = 0; // private by default in the for-loop
    #pragma omp for
    for(alignment_index=0; alignment_index < totalAlignments; ++alignment_index)
    {
      auto  current_read = *(read_sequence_ptr+alignment_index);
      auto  current_contig = *(contig_sequence_ptr+alignment_index);


      //TODO: THIS IS A BLOB THAT DOES 1 ALIGNMENT, IT SHOULD BE THE "WORK_STEAL KERNEL" FUNCTION
      const int32_t current_read_length = current_read.length();
      const int32_t current_contig_length = current_contig.length();

      //just convert the data to the format ssw wants...
      for(int i=0; i < current_read_length; ++i)
      {
        current_read_numeric[i] = table[(int)current_read[i]];
      }
      for(int i=0; i < current_contig_length; ++i)
      {
        current_contig_numeric[i] = table[(int)current_contig[i]];
      }

      //create a ssw profile from init
		  s_profile* p; //, *p_rc = 0; //we don't need the reverse compliment i assume.
      s_align* result;
      int32_t maskLen = current_read_length/2; //following the example, this is the mask length, used for suboptimal alignments ??

      //s_profile* ssw_init (const int8_t* read, const int32_t readLen, const int8_t* mat, const int32_t n, const int8_t score_size);
      // score_size = 2 means i dont know,,, 0 < 255, 1 >= 255, both are meaningless to me.
      p = ssw_init(current_read_numeric,current_read_length,mat,n,2);
			result = ssw_align (p, current_contig_numeric, current_contig_length, startGap * -1, extendGap * -1, flag, filter, 0, maskLen);

      //assign the results to the global array, no one shares an alignment index so it should be thread safe....
      alignments->query_begin[alignment_index] = result->ref_begin1;
      alignments->query_end[alignment_index] = result->ref_end1;
      alignments->ref_begin[alignment_index] = result->read_begin1;
      alignments->ref_end[alignment_index] = result->read_end1;
      alignments->top_scores[alignment_index] = result->score1;

      //destroy the result since ssw_init and ssw_align allocates something..
      align_destroy(result);
      init_destroy(p);

      //end TODO.

    }


    free(current_read_numeric);
    free(current_contig_numeric);

    } //end parallel region
    auto                          end  = NOW;
    std::chrono::duration<double> diff = end - start;
    std::cout << "Total Alignments:"<<totalAlignments<<"\n"<<"Max Reference Size:"<<maxContigSize<<"\n"<<"Max Query Size:"<<maxReadSize<<"\n" <<"Total Execution Time (seconds):"<< diff.count() <<std::endl;

    free(mata);
}


void
gpu_bsw_driver::kernel_driver_dna(std::vector<std::string> reads, std::vector<std::string> contigs, gpu_bsw_driver::alignment_results *alignments, short scores[4], float factor)
{
    short matchScore = scores[0], misMatchScore = scores[1], startGap = scores[2], extendGap = scores[3];
    unsigned maxContigSize = getMaxLength(contigs);
    unsigned maxReadSize = getMaxLength(reads);
    unsigned totalAlignments = contigs.size(); // assuming that read and contig vectors are same length

    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    omp_set_num_threads(deviceCount);
    std::cout<<"Number of available GPUs:"<<deviceCount<<"\n";

    unsigned NBLOCKS             = totalAlignments;
    unsigned alignmentsPerDevice = NBLOCKS / deviceCount;
    unsigned leftOver_device     = NBLOCKS % deviceCount;
    unsigned max_per_device = alignmentsPerDevice + leftOver_device;

    initialize_alignments(alignments, totalAlignments); // pinned memory allocation
    auto start = NOW;

    size_t tot_mem_req_per_aln = maxReadSize + maxContigSize + 2 * sizeof(int) + 5 * sizeof(short);
    
    #pragma omp parallel
    {
      int my_cpu_id = omp_get_thread_num();
      cudaSetDevice(my_cpu_id);
      int myGPUid;
      cudaGetDevice(&myGPUid);
      float total_time_cpu = 0;
      cudaStream_t streams_cuda[NSTREAMS];
      for(int stm = 0; stm < NSTREAMS; stm++){
        cudaStreamCreate(&streams_cuda[stm]);
      }

        if(my_cpu_id == 0)std::cout<<"Number of GPUs being used:"<<omp_get_num_threads()<<"\n";
        size_t gpu_mem_avail = get_tot_gpu_mem(myGPUid);
        unsigned max_alns_gpu = floor(((double)gpu_mem_avail*factor)/tot_mem_req_per_aln);
        unsigned max_alns_sugg = 20000;
        max_alns_gpu = max_alns_gpu > max_alns_sugg ? max_alns_sugg : max_alns_gpu;
        int       its    = (max_per_device>max_alns_gpu)?(ceil((double)max_per_device/max_alns_gpu)):1;
        std::cout<<"Mem (bytes) avail on device "<<myGPUid<<":"<<(long unsigned)gpu_mem_avail<<"\n";
        std::cout<<"Mem (bytes) using on device "<<myGPUid<<":"<<(long unsigned)gpu_mem_avail*factor<<"\n";

        int BLOCKS_l = alignmentsPerDevice;  //BW NOTE: alignments per device is the total number of alignments/device count. the last device gets all the carry over.
        if(my_cpu_id == deviceCount - 1)     //         the next few variables reflects that.. the amount of gpu data allocated is the most it will do per iteration.
            BLOCKS_l += leftOver_device;
        unsigned leftOvers    = BLOCKS_l % its;
        unsigned stringsPerIt = BLOCKS_l / its;
        gpu_alignments gpu_data(stringsPerIt + leftOvers); // gpu mallocs

        //BW TODO: these pointers to the results need to be controlled by the atomic queue
        //         instead, these will be scoped into the loop below...

        short* alAbeg = alignments->ref_begin + my_cpu_id * alignmentsPerDevice;
        short* alBbeg = alignments->query_begin + my_cpu_id * alignmentsPerDevice;
        short* alAend = alignments->ref_end + my_cpu_id * alignmentsPerDevice;
        short* alBend = alignments->query_end + my_cpu_id * alignmentsPerDevice;  // memory on CPU for copying the results
        short* top_scores_cpu = alignments->top_scores + my_cpu_id * alignmentsPerDevice;
        unsigned* offsetA_h;// = new unsigned[stringsPerIt + leftOvers];
        cudaMallocHost(&offsetA_h, sizeof(int)*(stringsPerIt + leftOvers));
        unsigned* offsetB_h;// = new unsigned[stringsPerIt + leftOvers];
        cudaMallocHost(&offsetB_h, sizeof(int)*(stringsPerIt + leftOvers));

        char *strA_d, *strB_d;
        cudaErrchk(cudaMalloc(&strA_d, maxContigSize * (stringsPerIt + leftOvers) * sizeof(char)));
        cudaErrchk(cudaMalloc(&strB_d, maxReadSize *(stringsPerIt + leftOvers)* sizeof(char)));

        char* strA;
        cudaMallocHost(&strA, sizeof(char)*maxContigSize * (stringsPerIt + leftOvers));
        char* strB;
        cudaMallocHost(&strB, sizeof(char)* maxReadSize *(stringsPerIt + leftOvers));

        float total_packing = 0;

        auto start2 = NOW;

        //BW NOTE: "its" will be very small in testing, it is iterating over the max alignments per device, that is,
        //         each kernel launch is a very large number of alignments, <20,000 by the hard coded value above.
        //         stringsPerIt is the total number of alignments divided by the number of iterations we will be looping through

        //          in practice, there will be several million alignments

        for(int perGPUIts = 0; perGPUIts < its; perGPUIts++)
        {
            auto packing_start = NOW;
            int                                      blocksLaunched = 0;
            std::vector<std::string>::const_iterator beginAVec;
            std::vector<std::string>::const_iterator endAVec;
            std::vector<std::string>::const_iterator beginBVec;
            std::vector<std::string>::const_iterator endBVec;
            if(perGPUIts == its - 1)
            {
                beginAVec = contigs.begin() + ((alignmentsPerDevice * my_cpu_id) + perGPUIts * stringsPerIt);
                endAVec = contigs.begin() + ((alignmentsPerDevice * my_cpu_id) + (perGPUIts + 1) * stringsPerIt) + leftOvers;  // so that each openmp thread has a copy of strings it needs to align
                beginBVec = reads.begin() + ((alignmentsPerDevice * my_cpu_id) + perGPUIts * stringsPerIt);
                endBVec = reads.begin() + ((alignmentsPerDevice * my_cpu_id) + (perGPUIts + 1) * stringsPerIt) + leftOvers;  // so that each openmp thread has a copy of strings it needs to align
                blocksLaunched = stringsPerIt + leftOvers;
            }
            else
            {
                beginAVec = contigs.begin() + ((alignmentsPerDevice * my_cpu_id) + perGPUIts * stringsPerIt);
                endAVec = contigs.begin() + (alignmentsPerDevice * my_cpu_id) + (perGPUIts + 1) * stringsPerIt; // so that each openmp thread has a copy of strings it needs to align
                beginBVec = reads.begin() + ((alignmentsPerDevice * my_cpu_id) + perGPUIts * stringsPerIt);
                endBVec = reads.begin() + (alignmentsPerDevice * my_cpu_id) +  (perGPUIts + 1) * stringsPerIt;  // so that each openmp thread has a copy of strings it needs to align
                blocksLaunched = stringsPerIt;
            }


            //BW TODO: Remove ABOVE, what we really want to do is
            //         request a batch of work from the atomic queue
            //         which should hold similar information, but really
            //         just the pointers for beginAVec/endAVec, beginBVec/endBVec
            //         as well as the results

            //         sequencesA and sequencesB is always a subset of the input contigs/reads.

            std::vector<std::string> sequencesA(beginAVec, endAVec);
            std::vector<std::string> sequencesB(beginBVec, endBVec);
            long unsigned running_sum = 0;
            int sequences_per_stream = (blocksLaunched) / NSTREAMS;
            int sequences_stream_leftover = (blocksLaunched) % NSTREAMS;
            long unsigned half_length_A = 0;
            long unsigned half_length_B = 0;

            auto start_cpu = NOW;

            //BW NOTE: the above basically "splits out" the input data per device, then per iteration
            //         per iteration gets split up per stream, but there are just 2.

            //BW NOTE: offsetA/B_h, is an "offset array", this is essentially a scan that is stored
            //         since we are always using 2 streams here, we actually "reset" this
            //         scan at the end of the number of sequences being sent through that stream
            //         there will also be a kernel launch per stream.
            //         its primary use is for the mem copy operations i think...

            for(int i = 0; i < sequencesA.size(); i++)
            {
                running_sum +=sequencesA[i].size();
                offsetA_h[i] = running_sum;//sequencesA[i].size();
                if(i == sequences_per_stream - 1){
                    half_length_A = running_sum;
                    running_sum = 0;
                  }
            }
            long unsigned totalLengthA = half_length_A + offsetA_h[sequencesA.size() - 1];

            running_sum = 0;
            for(int i = 0; i < sequencesB.size(); i++)
            {
                running_sum +=sequencesB[i].size();
                offsetB_h[i] = running_sum; //sequencesB[i].size();
                if(i == sequences_per_stream - 1){
                  half_length_B = running_sum;
                  running_sum = 0;
                }
            }
            long unsigned totalLengthB = half_length_B + offsetB_h[sequencesB.size() - 1];

            auto end_cpu = NOW;
            std::chrono::duration<double> cpu_dur = end_cpu - start_cpu;

            total_time_cpu += cpu_dur.count();
            long unsigned offsetSumA = 0;
            long unsigned offsetSumB = 0;

            //BW NOTE: strA and strB live in host memory, per thread, this is copying the sequences to there.
            //         sequencesA and sequencesB was constructed at the beginning of the loop, it is all the 
            //         query/ref sequences per iteration...

            for(int i = 0; i < sequencesA.size(); i++)
            {
                char* seqptrA = strA + offsetSumA;
                memcpy(seqptrA, sequencesA[i].c_str(), sequencesA[i].size());
                char* seqptrB = strB + offsetSumB;
                memcpy(seqptrB, sequencesB[i].c_str(), sequencesB[i].size());
                offsetSumA += sequencesA[i].size();
                offsetSumB += sequencesB[i].size();
            }

            auto packing_end = NOW;
            std::chrono::duration<double> packing_dur = packing_end - packing_start;

            total_packing += packing_dur.count();

            asynch_mem_copies_htd(&gpu_data, offsetA_h, offsetB_h, strA, strA_d, strB, strB_d, half_length_A, half_length_B, totalLengthA, totalLengthB, sequences_per_stream, sequences_stream_leftover, streams_cuda);


            //BW NOTE: minSize is the lesser of the biggest query or reference string.
            unsigned minSize = (maxReadSize < maxContigSize) ? maxReadSize : maxContigSize;
            unsigned totShmem = 3 * (minSize + 1) * sizeof(short);
            unsigned alignmentPad = 4 + (4 - totShmem % 4);
            size_t   ShmemBytes = totShmem + alignmentPad;
            if(ShmemBytes > 48000)
                cudaFuncSetAttribute(gpu_bsw::sequence_dna_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, ShmemBytes);

            gpu_bsw::sequence_dna_kernel<<<sequences_per_stream, minSize, ShmemBytes, streams_cuda[0]>>>(
                strA_d, strB_d, gpu_data.offset_ref_gpu, gpu_data.offset_query_gpu, gpu_data.ref_start_gpu,
                gpu_data.ref_end_gpu, gpu_data.query_start_gpu, gpu_data.query_end_gpu, gpu_data.scores_gpu, matchScore, misMatchScore, startGap, extendGap);
            cudaErrchk(cudaGetLastError());

            gpu_bsw::sequence_dna_kernel<<<sequences_per_stream + sequences_stream_leftover, minSize, ShmemBytes, streams_cuda[1]>>>(
                strA_d + half_length_A, strB_d + half_length_B, gpu_data.offset_ref_gpu + sequences_per_stream, gpu_data.offset_query_gpu + sequences_per_stream,
                 gpu_data.ref_start_gpu + sequences_per_stream, gpu_data.ref_end_gpu + sequences_per_stream, gpu_data.query_start_gpu + sequences_per_stream, gpu_data.query_end_gpu + sequences_per_stream,
                 gpu_data.scores_gpu + sequences_per_stream, matchScore, misMatchScore, startGap, extendGap);
            cudaErrchk(cudaGetLastError());

            // copyin back end index so that we can find new min
            asynch_mem_copies_dth_mid(&gpu_data, alAend, alBend, sequences_per_stream, sequences_stream_leftover, streams_cuda);

            cudaStreamSynchronize (streams_cuda[0]);
            cudaStreamSynchronize (streams_cuda[1]);

            auto sec_cpu_start = NOW;
            int newMin = get_new_min_length(alAend, alBend, blocksLaunched); // find the new largest of smaller lengths
            auto sec_cpu_end = NOW;
            std::chrono::duration<double> dur_sec_cpu = sec_cpu_end - sec_cpu_start;
            total_time_cpu += dur_sec_cpu.count();

            gpu_bsw::sequence_dna_reverse<<<sequences_per_stream, newMin, ShmemBytes, streams_cuda[0]>>>(
                    strA_d, strB_d, gpu_data.offset_ref_gpu, gpu_data.offset_query_gpu, gpu_data.ref_start_gpu,
                    gpu_data.ref_end_gpu, gpu_data.query_start_gpu, gpu_data.query_end_gpu, gpu_data.scores_gpu, matchScore, misMatchScore, startGap, extendGap);
            cudaErrchk(cudaGetLastError());

            gpu_bsw::sequence_dna_reverse<<<sequences_per_stream + sequences_stream_leftover, newMin, ShmemBytes, streams_cuda[1]>>>(
                    strA_d + half_length_A, strB_d + half_length_B, gpu_data.offset_ref_gpu + sequences_per_stream, gpu_data.offset_query_gpu + sequences_per_stream ,
                    gpu_data.ref_start_gpu + sequences_per_stream, gpu_data.ref_end_gpu + sequences_per_stream, gpu_data.query_start_gpu + sequences_per_stream, gpu_data.query_end_gpu + sequences_per_stream,
                    gpu_data.scores_gpu + sequences_per_stream, matchScore, misMatchScore, startGap, extendGap);
            cudaErrchk(cudaGetLastError());

            asynch_mem_copies_dth(&gpu_data, alAbeg, alBbeg, top_scores_cpu, sequences_per_stream, sequences_stream_leftover, streams_cuda);


            //BW NOTE: ^^^ I think the biggest danger is just writing the results to the wrong place, but it is going to be basically pointers and counts to contigous memory above
            //             so I think it's pretty safe to leave everything the same, i think the batch has to be less than 40k ?

            //BW TODO: REMOVE THESE INCREMENTS. you also kind of know what they are implicitly, ^^ alaBeg + stringsPerIt * perGPUIts, without modifying where the pointers are pointing at.
            //         What we want actually is to just check if there is more work to do via query to the atomic queue... can be part of a "while" condition at the top ofc.
                 alAbeg += stringsPerIt;
                 alBbeg += stringsPerIt;
                 alAend += stringsPerIt;
                 alBend += stringsPerIt;
                 top_scores_cpu += stringsPerIt;

		 cudaStreamSynchronize (streams_cuda[0]);
                 cudaStreamSynchronize (streams_cuda[1]);

        }  // for iterations end here

        auto end1  = NOW;
        std::chrono::duration<double> diff2 = end1 - start2;
        cudaErrchk(cudaFree(strA_d));
        cudaErrchk(cudaFree(strB_d));
        cudaFreeHost(offsetA_h);
        cudaFreeHost(offsetB_h);
        cudaFreeHost(strA);
        cudaFreeHost(strB);

        for(int i = 0; i < NSTREAMS; i++)
          cudaStreamDestroy(streams_cuda[i]);

        std::cout <<"cpu time:"<<total_time_cpu<<std::endl;
        std::cout <<"packing time:"<<total_packing<<std::endl;
        #pragma omp barrier
    }  // paralle pragma ends

    auto                          end  = NOW;
    std::chrono::duration<double> diff = end - start;
    std::cout << "Total Alignments:"<<totalAlignments<<"\n"<<"Max Reference Size:"<<maxContigSize<<"\n"<<"Max Query Size:"<<maxReadSize<<"\n" <<"Total Execution Time (seconds):"<< diff.count() <<std::endl;
}// end of DNA kernel

void
gpu_bsw_driver::kernel_driver_aa(std::vector<std::string> reads, std::vector<std::string> contigs, gpu_bsw_driver::alignment_results *alignments, short scoring_matrix[], short openGap, short extendGap, float factor)
{
    unsigned maxContigSize = getMaxLength(contigs);
    unsigned maxReadSize = getMaxLength(reads);
    unsigned totalAlignments = contigs.size(); // assuming that read and contig vectors are same length

    short encoding_matrix[] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                              0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                              0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                             23,0,0,0,0,0,0,0,0,0,0,0,0,0,
                             0,0,0,0,0,0,0,0,0,0,20,4,3,6,
                             13,7,8,9,0,11,10,12,2,0,14,5,
                             1,15,16,0,19,17,22,18,21};

    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    omp_set_num_threads(deviceCount);// one OMP thread per GPU
    std::cout<<"Number of available GPUs:"<<deviceCount<<"\n";

    cudaDeviceProp prop[deviceCount];
    for(int i = 0; i < deviceCount; i++)
      cudaGetDeviceProperties(&prop[i], 0);

    unsigned NBLOCKS             = totalAlignments;
    unsigned alignmentsPerDevice = NBLOCKS / deviceCount;
    unsigned leftOver_device     = NBLOCKS % deviceCount;
    unsigned max_per_device = alignmentsPerDevice + leftOver_device;
    
    initialize_alignments(alignments, totalAlignments); // pinned memory allocation
    auto start = NOW;
    size_t tot_mem_req_per_aln = maxReadSize + maxContigSize + 2 * sizeof(int) + 5 * sizeof(short);
    #pragma omp parallel
    {

      int my_cpu_id = omp_get_thread_num();
      cudaSetDevice(my_cpu_id);
      int myGPUid;
      cudaGetDevice(&myGPUid);
      float total_time_cpu = 0;
      cudaStream_t streams_cuda[NSTREAMS];
      for(int stm = 0; stm < NSTREAMS; stm++){
        cudaStreamCreate(&streams_cuda[stm]);
      }
      if(my_cpu_id == 0)std::cout<<"Number of GPUs being used:"<<omp_get_num_threads()<<"\n";
        size_t gpu_mem_avail = get_tot_gpu_mem(myGPUid);
        unsigned max_alns_gpu = floor(((double)gpu_mem_avail*factor)/tot_mem_req_per_aln);
        unsigned max_alns_sugg = 20000;
        max_alns_gpu = max_alns_gpu > max_alns_sugg ? max_alns_sugg : max_alns_gpu;
        int       its    = (max_per_device>max_alns_gpu)?(ceil((double)max_per_device/max_alns_gpu)):1;
        std::cout<<"Mem (bytes) avail on device "<<myGPUid<<":"<<(long unsigned)gpu_mem_avail<<"\n";
        std::cout<<"Mem (bytes) using on device "<<myGPUid<<":"<<(long unsigned)gpu_mem_avail*factor<<"\n";

      int BLOCKS_l = alignmentsPerDevice;
      if(my_cpu_id == deviceCount - 1)
          BLOCKS_l += leftOver_device;
      unsigned leftOvers    = BLOCKS_l % its;
      unsigned stringsPerIt = BLOCKS_l / its;
      gpu_alignments gpu_data(stringsPerIt + leftOvers); // gpu mallocs
      short *d_encoding_matrix, *d_scoring_matrix;
      cudaErrchk(cudaMalloc(&d_encoding_matrix, ENCOD_MAT_SIZE * sizeof(short)));
      cudaErrchk(cudaMalloc(&d_scoring_matrix, SCORE_MAT_SIZE * sizeof(short)));
      cudaErrchk(cudaMemcpy(d_encoding_matrix, encoding_matrix, ENCOD_MAT_SIZE * sizeof(short), cudaMemcpyHostToDevice));
      cudaErrchk(cudaMemcpy(d_scoring_matrix, scoring_matrix, SCORE_MAT_SIZE * sizeof(short), cudaMemcpyHostToDevice));

      short* alAbeg = alignments->ref_begin + my_cpu_id * alignmentsPerDevice;
      short* alBbeg = alignments->query_begin + my_cpu_id * alignmentsPerDevice;
      short* alAend = alignments->ref_end + my_cpu_id * alignmentsPerDevice;
      short* alBend = alignments->query_end + my_cpu_id * alignmentsPerDevice;  // memory on CPU for copying the results
      short* top_scores_cpu = alignments->top_scores + my_cpu_id * alignmentsPerDevice;

      unsigned* offsetA_h;// = new unsigned[stringsPerIt + leftOvers];
      cudaMallocHost(&offsetA_h, sizeof(int)*(stringsPerIt + leftOvers));
      unsigned* offsetB_h;// = new unsigned[stringsPerIt + leftOvers];
      cudaMallocHost(&offsetB_h, sizeof(int)*(stringsPerIt + leftOvers));

      char *strA_d, *strB_d;
      cudaErrchk(cudaMalloc(&strA_d, maxContigSize * (stringsPerIt + leftOvers) * sizeof(char)));
      cudaErrchk(cudaMalloc(&strB_d, maxReadSize *(stringsPerIt + leftOvers)* sizeof(char)));

      char* strA;
      cudaMallocHost(&strA, sizeof(char)*maxContigSize * (stringsPerIt + leftOvers));
      char* strB;
      cudaMallocHost(&strB, sizeof(char)* maxReadSize *(stringsPerIt + leftOvers));

      float total_packing = 0;

      auto start2 = NOW;
      std::cout<<"loop begin\n";
      for(int perGPUIts = 0; perGPUIts < its; perGPUIts++)
      {
          auto packing_start = NOW;
          int                                      blocksLaunched = 0;
          std::vector<std::string>::const_iterator beginAVec;
          std::vector<std::string>::const_iterator endAVec;
          std::vector<std::string>::const_iterator beginBVec;
          std::vector<std::string>::const_iterator endBVec;
          if(perGPUIts == its - 1)
          {
              beginAVec = contigs.begin() + ((alignmentsPerDevice * my_cpu_id) + perGPUIts * stringsPerIt);
              endAVec = contigs.begin() + ((alignmentsPerDevice * my_cpu_id) + (perGPUIts + 1) * stringsPerIt) + leftOvers;  // so that each openmp thread has a copy of strings it needs to align
              beginBVec = reads.begin() + ((alignmentsPerDevice * my_cpu_id) + perGPUIts * stringsPerIt);
              endBVec = reads.begin() + ((alignmentsPerDevice * my_cpu_id) + (perGPUIts + 1) * stringsPerIt) + leftOvers;  // so that each openmp thread has a copy of strings it needs to align
              blocksLaunched = stringsPerIt + leftOvers;
          }
          else
          {
              beginAVec = contigs.begin() + ((alignmentsPerDevice * my_cpu_id) + perGPUIts * stringsPerIt);
              endAVec = contigs.begin() + (alignmentsPerDevice * my_cpu_id) + (perGPUIts + 1) * stringsPerIt; // so that each openmp thread has a copy of strings it needs to align
              beginBVec = reads.begin() + ((alignmentsPerDevice * my_cpu_id) + perGPUIts * stringsPerIt);
              endBVec = reads.begin() + (alignmentsPerDevice * my_cpu_id) +  (perGPUIts + 1) * stringsPerIt;  // so that each openmp thread has a copy of strings it needs to align
              blocksLaunched = stringsPerIt;
          }

          std::vector<std::string> sequencesA(beginAVec, endAVec);
          std::vector<std::string> sequencesB(beginBVec, endBVec);
          unsigned running_sum = 0;
          int sequences_per_stream = (blocksLaunched) / NSTREAMS;
          int sequences_stream_leftover = (blocksLaunched) % NSTREAMS;
          unsigned half_length_A = 0;
          unsigned half_length_B = 0;

          auto start_cpu = NOW;

          for(int i = 0; i < sequencesA.size(); i++)
          {
              running_sum +=sequencesA[i].size();
              offsetA_h[i] = running_sum;//sequencesA[i].size();
              if(i == sequences_per_stream - 1){
                  half_length_A = running_sum;
                  running_sum = 0;
                }
          }
          unsigned totalLengthA = half_length_A + offsetA_h[sequencesA.size() - 1];

          running_sum = 0;
          for(int i = 0; i < sequencesB.size(); i++)
          {
              running_sum +=sequencesB[i].size();
              offsetB_h[i] = running_sum; //sequencesB[i].size();
              if(i == sequences_per_stream - 1){
                half_length_B = running_sum;
                running_sum = 0;
              }
          }
          unsigned totalLengthB = half_length_B + offsetB_h[sequencesB.size() - 1];

          auto end_cpu = NOW;
          std::chrono::duration<double> cpu_dur = end_cpu - start_cpu;

          total_time_cpu += cpu_dur.count();
          unsigned offsetSumA = 0;
          unsigned offsetSumB = 0;

          for(int i = 0; i < sequencesA.size(); i++)
          {
              char* seqptrA = strA + offsetSumA;
              memcpy(seqptrA, sequencesA[i].c_str(), sequencesA[i].size());
              char* seqptrB = strB + offsetSumB;
              memcpy(seqptrB, sequencesB[i].c_str(), sequencesB[i].size());
              offsetSumA += sequencesA[i].size();
              offsetSumB += sequencesB[i].size();
          }

          auto packing_end = NOW;
          std::chrono::duration<double> packing_dur = packing_end - packing_start;

          total_packing += packing_dur.count();

          asynch_mem_copies_htd(&gpu_data, offsetA_h, offsetB_h, strA, strA_d, strB, strB_d, half_length_A, half_length_B, totalLengthA, totalLengthB, sequences_per_stream, sequences_stream_leftover, streams_cuda);
          unsigned minSize = (maxReadSize < maxContigSize) ? maxReadSize : maxContigSize;
          unsigned totShmem = 3 * (minSize + 1) * sizeof(short);
          unsigned alignmentPad = 4 + (4 - totShmem % 4);
          size_t   ShmemBytes = totShmem + alignmentPad;
          if(ShmemBytes > 48000)
              cudaFuncSetAttribute(gpu_bsw::sequence_dna_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, ShmemBytes);

          gpu_bsw::sequence_aa_kernel<<<sequences_per_stream, minSize, ShmemBytes, streams_cuda[0]>>>(
              strA_d, strB_d, gpu_data.offset_ref_gpu, gpu_data.offset_query_gpu, gpu_data.ref_start_gpu,
              gpu_data.ref_end_gpu, gpu_data.query_start_gpu, gpu_data.query_end_gpu, gpu_data.scores_gpu,
              openGap, extendGap, d_scoring_matrix, d_encoding_matrix);
          cudaErrchk(cudaGetLastError());

          gpu_bsw::sequence_aa_kernel<<<sequences_per_stream + sequences_stream_leftover, minSize, ShmemBytes, streams_cuda[1]>>>(
              strA_d + half_length_A, strB_d + half_length_B, gpu_data.offset_ref_gpu + sequences_per_stream, gpu_data.offset_query_gpu + sequences_per_stream,
                gpu_data.ref_start_gpu + sequences_per_stream, gpu_data.ref_end_gpu + sequences_per_stream, gpu_data.query_start_gpu + sequences_per_stream, gpu_data.query_end_gpu + sequences_per_stream,
                gpu_data.scores_gpu + sequences_per_stream, openGap, extendGap, d_scoring_matrix, d_encoding_matrix);
          cudaErrchk(cudaGetLastError());

          // copyin back end index so that we can find new min
          asynch_mem_copies_dth_mid(&gpu_data, alAend, alBend, sequences_per_stream, sequences_stream_leftover, streams_cuda);

          cudaStreamSynchronize (streams_cuda[0]);
          cudaStreamSynchronize (streams_cuda[1]);

          auto sec_cpu_start = NOW;
          int newMin = get_new_min_length(alAend, alBend, blocksLaunched); // find the new largest of smaller lengths
          auto sec_cpu_end = NOW;
          std::chrono::duration<double> dur_sec_cpu = sec_cpu_end - sec_cpu_start;
          total_time_cpu += dur_sec_cpu.count();

          gpu_bsw::sequence_aa_reverse<<<sequences_per_stream, newMin, ShmemBytes, streams_cuda[0]>>>(
                  strA_d, strB_d, gpu_data.offset_ref_gpu, gpu_data.offset_query_gpu, gpu_data.ref_start_gpu,
                  gpu_data.ref_end_gpu, gpu_data.query_start_gpu, gpu_data.query_end_gpu, gpu_data.scores_gpu, openGap, extendGap, d_scoring_matrix, d_encoding_matrix);
          cudaErrchk(cudaGetLastError());

          gpu_bsw::sequence_aa_reverse<<<sequences_per_stream + sequences_stream_leftover, newMin, ShmemBytes, streams_cuda[1]>>>(
                  strA_d + half_length_A, strB_d + half_length_B, gpu_data.offset_ref_gpu + sequences_per_stream, gpu_data.offset_query_gpu + sequences_per_stream ,
                  gpu_data.ref_start_gpu + sequences_per_stream, gpu_data.ref_end_gpu + sequences_per_stream, gpu_data.query_start_gpu + sequences_per_stream, gpu_data.query_end_gpu + sequences_per_stream,
                  gpu_data.scores_gpu + sequences_per_stream, openGap, extendGap, d_scoring_matrix, d_encoding_matrix);
          cudaErrchk(cudaGetLastError());

          asynch_mem_copies_dth(&gpu_data, alAbeg, alBbeg, top_scores_cpu, sequences_per_stream, sequences_stream_leftover, streams_cuda);

                alAbeg += stringsPerIt;
                alBbeg += stringsPerIt;
                alAend += stringsPerIt;
                alBend += stringsPerIt;
                top_scores_cpu += stringsPerIt;
		
		 cudaStreamSynchronize (streams_cuda[0]);
                 cudaStreamSynchronize (streams_cuda[1]);

      }  // for iterations end here

        auto end1  = NOW;
        std::chrono::duration<double> diff2 = end1 - start2;
        cudaErrchk(cudaFree(strA_d));
        cudaErrchk(cudaFree(strB_d));
        cudaFreeHost(offsetA_h);
        cudaFreeHost(offsetB_h);
        cudaFreeHost(strA);
        cudaFreeHost(strB);

        for(int i = 0; i < NSTREAMS; i++)
          cudaStreamDestroy(streams_cuda[i]);

        std::cout <<"cpu time:"<<total_time_cpu<<std::endl;
        std::cout <<"packing time:"<<total_packing<<std::endl;
        #pragma omp barrier
    }  // paralle pragma ends
    auto                          end  = NOW;
    std::chrono::duration<double> diff = end - start;
    std::cout << "Total Alignments:"<<totalAlignments<<"\n"<<"Max Reference Size:"<<maxContigSize<<"\n"<<"Max Query Size:"<<maxReadSize<<"\n" <<"Total Execution Time (seconds):"<< diff.count() <<std::endl;
}// end of amino acids kernel