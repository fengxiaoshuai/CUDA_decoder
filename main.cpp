#include "decoder.h"
#include "decoding.h"
#include <cstdio>
#include <vector>
#include <string>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sys/time.h>

using namespace fastertransformer;
using namespace std;


void load(vector<float>& weight, string dir)
{
        ifstream input(dir);

        if (input.fail())
        {
                cout << "File does not exist" << endl;
                cout << "Exit program" << endl;
                return;
        }
        float num=0.0;
        while (input>>num)  // 当没有读到文件结尾
        {
                weight.push_back(num);
                //cout << num << endl;
        }
        input.close();

}

void load_layer_weight(vector<vector<float>>& layer_weight, int num)
{
        cout << "start read layer " << num << " weight" << endl;
        vector<float> layer_self_scale;//0
        vector<float> layer_self_bias;//1
        vector<float> layer_self_q;//2
        vector<float> layer_self_k;//3
        vector<float> layer_self_v;//4
        vector<float> layer_self_last;//5

        vector<float> layer_encdec_scale;//6
        vector<float> layer_encdec_bias;//7
        vector<float> layer_encdec_q;//8
        vector<float> layer_encdec_k;//cache k//9
        vector<float> layer_encdec_v;//cache v//10
        vector<float> layer_encdec_last;//11

        vector<float> layer_ffn_scale;//12
        vector<float> layer_ffn_bias;//13
        vector<float> layer_ffn_first_weight;//14
        vector<float> layer_ffn_first_bias;//15
        vector<float> layer_ffn_second_weight;//16
        vector<float> layer_ffn_second_bias;//17

        vector<float> layer_self_position_key;//18
        vector<float> layer_self_position_value;//19


        cout << "...:load self attention weight" << endl;
        string name = "./weight/layer_" + to_string(num) ;
        load(layer_self_scale, name + "_self_scale.txt");
        load(layer_self_bias, name + "_self_bias.txt");
        load(layer_self_q, name + "_self_q.txt");
        load(layer_self_k, name + "_self_k.txt");
        load(layer_self_v, name + "_self_v.txt");
        load(layer_self_last, name + "_self_last.txt");
        load(layer_self_position_key, name + "_self_position_key.txt");
        load(layer_self_position_value, name + "_self_position_value.txt");
        cout << "...:load encdec attention weight" << endl;
        load(layer_encdec_scale, name + "_encdec_scale.txt");
        load(layer_encdec_bias, name + "_encdec_bias.txt");
        load(layer_encdec_q, name + "_encdec_q.txt");
        load(layer_encdec_k, name + "_encdec_k.txt");
        load(layer_encdec_v, name + "_encdec_v.txt");
        load(layer_encdec_last, name + "_encdec_last.txt");
        cout << "...:load read fnn weight" << endl;
        load(layer_ffn_scale, name + "_ffn_scale.txt");
        load(layer_ffn_bias, name + "_ffn_bias.txt");
        load(layer_ffn_first_weight, name + "_ffn_first_weight.txt");
        load(layer_ffn_first_bias, name + "_ffn_first_bias.txt");
        load(layer_ffn_second_weight, name + "_ffn_second_weight.txt");
        load(layer_ffn_second_bias, name + "_ffn_second_bias.txt");


        layer_weight.push_back(layer_self_scale);
        layer_weight.push_back(layer_self_bias);
        layer_weight.push_back(layer_self_q);
        layer_weight.push_back(layer_self_k);
        layer_weight.push_back(layer_self_v);
        layer_weight.push_back(layer_self_last);

        layer_weight.push_back(layer_encdec_scale);
        layer_weight.push_back(layer_encdec_bias);
        layer_weight.push_back(layer_encdec_q);
        layer_weight.push_back(layer_encdec_k);
        layer_weight.push_back(layer_encdec_v);
        layer_weight.push_back(layer_encdec_last);
        layer_weight.push_back(layer_ffn_scale);
        layer_weight.push_back(layer_ffn_bias);
        layer_weight.push_back(layer_ffn_first_weight);
        layer_weight.push_back(layer_ffn_first_bias);
        layer_weight.push_back(layer_ffn_second_weight);
        layer_weight.push_back(layer_ffn_second_bias);

        layer_weight.push_back(layer_self_position_key);
        layer_weight.push_back(layer_self_position_value);

        cout << "...:end layer " << num << " weight" << endl;
}


template<typename T>
void BuildBias(const int& batch_size, const int& length,  int* mask, T* bias)
{
        for (int i = 0; i < batch_size*length; i++)
        {
                bias[i] *= (1-mask[i]);
        }
}

template<typename T>
void device_malloc(T** ptr, int size, T* h_ptr)
{
  check_cuda_error(cudaMalloc((void**)ptr, size));
  check_cuda_error(cudaMemcpy(*ptr, h_ptr, size, cudaMemcpyHostToDevice));
}

template<typename T>
void decoding_sample(int batch_size,
                     int head_num,
                     int size_per_head,
                     int vocab_size,
                     int length,
                     int decoder_layers,
                     int hidden_unit,
                     int decode_length,
                     int language_num,
                     int* language_id,
                     int* mask)
{
  const int max_decode_length = decode_length;
  const int seq_len = length;
  const int end_id = 2;
  const int hidden_units = head_num * size_per_head;
  const int inner_size = 4096;
  const int max_position = 20;
  int* start_id = language_id;

  vector<float> encode_out;
  load(encode_out, "./weight/encode_out.txt");

  cout << "start load embedding" << endl;
  vector<float> weight_embedding;
  load(weight_embedding, "./weight/embedding.txt");
  cout<<"weight_embedding: "<<weight_embedding.size()<<endl;
  vector<float> language_embedding;
  load(language_embedding, "./weight/language_embedding.txt");
  cout << "end load embedding" << endl;

  vector<float> weight_scale;
  load(weight_scale, "./weight/scale.txt");
  vector<float> weight_bias;
  load(weight_bias, "./weight/bias.txt");
  float* d_test;
  device_malloc(&d_test, 1024, weight_bias.data());

  cout << "start load logits" << endl;
  vector<float> logit_weight;
  load(logit_weight, "./weight/logit.txt");
  cout << "end load logits" << endl;

  vector<vector<vector<float>>> weight(decoder_layers);
  for(int i = 0; i<decoder_layers; i++)
  {
        load_layer_weight(weight[i], i);
  }


  cublasHandle_t cublasHandle;
  check_cuda_error(cublasCreate(&cublasHandle));

  cudaStream_t stream;
  check_cuda_error(cudaStreamCreate(&stream));
  check_cuda_error(cublasSetStream(cublasHandle, stream));

  fastertransformer::Allocator<AllocatorType::CUDA> allocator(0);
  DecoderInitParam<T> *param = new DecoderInitParam<T>[decoder_layers];

  T h_bias[batch_size * seq_len] = {-1e9};
  for(int i=0; i<batch_size*seq_len; i++)
  {
    h_bias[i] = -1e9;
  }
  BuildBias(batch_size, length, mask, h_bias);

  cout << "start malloc for GPU" << endl;
  for(int i = 0; i < decoder_layers; i++)
  {
    param[i].stream = stream;
    param[i].cublas_handle = cublasHandle;

    T *d_self_Q_kernel, *d_self_K_kernel, *d_self_V_kernel, *d_self_output_kernel, *d_self_gamma, *d_self_beta;
    T *d_self_position_key, *d_self_position_value;
    T *d_cross_Q_kernel, *d_cross_K_kernel, *d_cross_V_kernel, *d_cross_output_kernel,*d_cross_bias_kernel, *d_cross_gamma, *d_cross_beta;
    T *d_ffn_kernel1, *d_ffn_bias1, *d_ffn_kernel2, *d_ffn_bias2, *d_ffn_gamma, *d_ffn_beta;

    device_malloc(&d_self_gamma, sizeof(T) * hidden_units, weight[i][0].data());
    device_malloc(&d_self_beta, sizeof(T) * hidden_units, weight[i][1].data());
    device_malloc(&d_self_Q_kernel, sizeof(T) * hidden_units * hidden_units, weight[i][2].data());
    device_malloc(&d_self_K_kernel, sizeof(T) * hidden_units * hidden_units, weight[i][3].data());
    device_malloc(&d_self_V_kernel, sizeof(T) * hidden_units * hidden_units, weight[i][4].data());
    device_malloc(&d_self_output_kernel, sizeof(T) * hidden_units * hidden_units, weight[i][5].data());
    device_malloc(&d_self_position_key, sizeof(T) * (max_position*2+1) * size_per_head, weight[i][18].data());
    device_malloc(&d_self_position_value, sizeof(T) * (max_position*2+1) * size_per_head, weight[i][19].data());

    device_malloc(&d_cross_gamma, sizeof(T) * hidden_units, weight[i][6].data());
    device_malloc(&d_cross_beta, sizeof(T) * hidden_units, weight[i][7].data());
    device_malloc(&d_cross_Q_kernel, sizeof(T) * hidden_units * hidden_units, weight[i][8].data());
    device_malloc(&d_cross_K_kernel, sizeof(T) * hidden_unit * hidden_units, weight[i][9].data());
    device_malloc(&d_cross_V_kernel, sizeof(T) * hidden_unit * hidden_units, weight[i][10].data());
    device_malloc(&d_cross_output_kernel, sizeof(T) * hidden_units * hidden_units, weight[i][11].data());
    device_malloc(&d_cross_bias_kernel, sizeof(T) * batch_size * seq_len, h_bias);

    device_malloc(&d_ffn_gamma, sizeof(T) * hidden_units, weight[i][12].data());
    device_malloc(&d_ffn_beta, sizeof(T) * hidden_units, weight[i][13].data());
    device_malloc(&d_ffn_kernel1, sizeof(T) * inner_size * hidden_units, weight[i][14].data());
    device_malloc(&d_ffn_bias1, sizeof(T) * inner_size, weight[i][15].data());
    device_malloc(&d_ffn_kernel2, sizeof(T) * inner_size * hidden_units, weight[i][16].data());
    device_malloc(&d_ffn_bias2, sizeof(T) * hidden_units, weight[i][17].data());


    param[i].self_layernorm.gamma = d_self_gamma;
    param[i].self_layernorm.beta = d_self_beta;
    param[i].self_attention.query_weight = d_self_Q_kernel;
    param[i].self_attention.key_weight = d_self_K_kernel;
    param[i].self_attention.value_weight = d_self_V_kernel;
    param[i].self_attention.attention_output_weight = d_self_output_kernel;
    param[i].self_attention.position_key = d_self_position_key;
    param[i].self_attention.position_value = d_self_position_value;

    param[i].cross_layernorm.gamma = d_cross_gamma;
    param[i].cross_layernorm.beta = d_cross_beta;
    param[i].cross_attention.query_weight = d_cross_Q_kernel;
    param[i].cross_attention.key_weight = d_cross_K_kernel;
    param[i].cross_attention.value_weight = d_cross_V_kernel;
    param[i].cross_attention.attention_output_weight = d_cross_output_kernel;
    param[i].cross_bias = d_cross_bias_kernel;

    param[i].ffn_layernorm.gamma = d_ffn_gamma;
    param[i].ffn_layernorm.beta = d_ffn_beta;
    param[i].ffn.intermediate_weight.kernel = d_ffn_kernel1;
    param[i].ffn.intermediate_weight.bias = d_ffn_bias1;
    param[i].ffn.output_weight.kernel = d_ffn_kernel2;
    param[i].ffn.output_weight.bias = d_ffn_bias2;
  }

  DecodingInitParam<T> decoding_params;

  T *d_encodeout_tensor;
  T *d_embedding_table_init;
  T *d_embedding_table_run;
  T *d_gamma;
  T *d_beta;
  T *d_embedding_kernel;
  int* d_start_ids;

  int* d_output_ids;
  int* d_sequence_lengths;
  device_malloc(&d_encodeout_tensor, sizeof(T) * hidden_units * seq_len * batch_size , encode_out.data());
  device_malloc(&d_embedding_table_init, sizeof(T) * language_num  *hidden_units , language_embedding.data());
  device_malloc(&d_embedding_table_run, sizeof(T) * vocab_size * hidden_units , weight_embedding.data());
  device_malloc(&d_gamma, sizeof(T) * hidden_units, weight_scale.data());
  device_malloc(&d_beta, sizeof(T) * hidden_units, weight_bias.data());
  device_malloc(&d_embedding_kernel, sizeof(T) * hidden_units * vocab_size, logit_weight.data());
  device_malloc(&d_start_ids, sizeof(int) * batch_size, start_id);

  check_cuda_error(cudaMalloc((void**)&d_output_ids, sizeof(int) * (max_decode_length) * batch_size ));

  decoding_params.cublas_handle = cublasHandle;
  decoding_params.stream = stream;
  decoding_params.memory_tensor = d_encodeout_tensor;
  decoding_params.embedding_table_init = d_embedding_table_init;
  decoding_params.embedding_table_run = d_embedding_table_run;
  decoding_params.embedding_kernel = d_embedding_kernel;
  decoding_params.output_ids = d_output_ids;
  decoding_params.layernorm.gamma = d_gamma;
  decoding_params.layernorm.beta = d_beta;

  const fastertransformer::OperationType type = sizeof(T) == sizeof(float) ? OperationType::FP32 : OperationType::FP16;

  cout << "end malloc for GPU" << endl;
 
  DecodingOpenNMT<type> *decoding = new DecodingOpenNMT<type>(allocator, batch_size,
                                         max_decode_length, head_num, size_per_head,
                                         vocab_size, decoder_layers,
                                         hidden_unit, seq_len,
                                         d_start_ids, end_id);

  //warm up
  int ite = 5;
  for(int i = 0; i < ite; ++i)
    decoding->forward(param, decoding_params);

  struct timeval start, end;
  cudaDeviceSynchronize();
  gettimeofday(&start, NULL);

  for(int i = 0; i < 10; ++i)
    decoding->forward(param, decoding_params);
  cudaDeviceSynchronize();

  gettimeofday(&end, NULL);
  printf("time: %.2f ms \n",((end.tv_sec - start.tv_sec) * 1000 + (end.tv_usec - start.tv_usec) * 0.001) / 10);
  delete decoding;
  return ;
}



int main(int argc, char* argv[])
{
  //show the properties of device
  struct cudaDeviceProp prop;
  check_cuda_error(cudaGetDeviceProperties(&prop, 0));
  printf("Device %s\n", prop.name);
  // need 10 para
  if(argc != 11)
  {
    printf("[ERROR] decoding_sample batch_size head_num size_per_head vocab_size seq_len num_layer hidden_unit decode_length  language_num is_fp16\n");
    printf("e.g. ./bin/decoding_sample 2 16 64 32768 17 6 1024 4 0 \n");
    return 0;
  }

  const int batch_size = atoi(argv[1]);
  const int head_num = atoi(argv[2]);
  const int size_per_head = atoi(argv[3]);
  const int vocab_size = atoi(argv[4]);
  const int length = atoi(argv[5]);
  const int decoder_layers = atoi(argv[6]);
  const int hidden_unit = atoi(argv[7]);
  const int decode_length = atoi(argv[8]);
  const int language_num = atoi(argv[9]);
  int language_id[batch_size] = {1,1};
  int mask[batch_size * length] = {1,1,1,1,1,0,0,0, 1,1,1,1,1,1,0,0};


  cout<<"batch_size: "<<batch_size<<endl;
  cout<<"head_num: "<<head_num<<endl;
  cout<<"size_per_head: "<<size_per_head<<endl;
  cout<<"vocab_size: "<<vocab_size<<endl;
  cout<<"length: "<<length<<endl;
  cout<<"decoder_layer: "<<decoder_layers<<endl;
  cout<<"hidden_unit: "<<hidden_unit<<endl;
  cout<<"decode_length: "<<decode_length<<endl;
  cout<<"language_num: "<<language_num<<endl;
  cout<<"language_id: ";
  for(int i=0; i<batch_size; i++)
  {
   cout<<language_id[i]<<" ";
  }
  cout<<endl;
  cout<<"mask: ";
  for(int i=0; i<batch_size * length; i++)
  {
   cout<<mask[i]<<" ";
  }
  cout<<endl;

  decoding_sample<float>(batch_size, head_num, size_per_head, vocab_size, length, decoder_layers, hidden_unit, decode_length, language_num, language_id, mask);

  return 0;
}
























