#ifndef EFANNA2E_UTIL_H
#define EFANNA2E_UTIL_H
#include <random>
#include <fstream>
#include <iostream>
#include <cstring>
#include <algorithm>
#ifdef __APPLE__
#else
#include <malloc.h>
#endif
namespace efanna2e {

    static void GenRandom(std::mt19937 &rng, unsigned *addr, unsigned size, unsigned N) {
        for (unsigned i = 0; i < size; ++i) {
            addr[i] = rng() % (N - size);
        }
        std::sort(addr, addr + size);
        for (unsigned i = 1; i < size; ++i) {
            if (addr[i] <= addr[i - 1]) {
                addr[i] = addr[i - 1] + 1;
            }
        }
        unsigned off = rng() % N;
        for (unsigned i = 0; i < size; ++i) {
            addr[i] = (addr[i] + off) % N;
        }
    }

    inline void load_data(char* filename, float*& data, unsigned& num,unsigned& dim){// load data with sift10K pattern
      std::ifstream in(filename, std::ios::binary);
      if(!in.is_open()){std::cout<<"open file error"<<std::endl;exit(-1);}
      in.read((char*)&dim,4);
      std::cout<<"data dimension: "<<dim<<std::endl;
      in.seekg(0,std::ios::end);
      std::ios::pos_type ss = in.tellg();
      size_t fsize = (size_t)ss;
      num = (unsigned)(fsize / (dim+1) / 4);
      data = new float[(uint64_t)num * (uint64_t)dim * sizeof(float)];

      in.seekg(0,std::ios::beg);
      for(size_t i = 0; i < num; i++){
        in.seekg(4,std::ios::cur);
        in.read((char*)(data+i*dim),dim*4);
      }
      in.close();
    }

    inline float* data_align(float* data_ori, unsigned point_num, unsigned& dim){
      #ifdef __GNUC__
      #ifdef __AVX__
        #define DATA_ALIGN_FACTOR 8
      #else
      #ifdef __SSE2__
        #define DATA_ALIGN_FACTOR 4
      #else
        #define DATA_ALIGN_FACTOR 1
      #endif
      #endif
      #endif

      std::cout << "align with : "<<DATA_ALIGN_FACTOR << std::endl;
      float* data_new=0;
      unsigned new_dim = (dim + DATA_ALIGN_FACTOR - 1) / DATA_ALIGN_FACTOR * DATA_ALIGN_FACTOR;
      std::cout << "align to new dim: "<<new_dim << std::endl;
      #ifdef __APPLE__
        data_new = new float[(uint64_t)new_dim * (uint64_t)point_num];
      #else
        data_new = (float*)memalign(DATA_ALIGN_FACTOR * 4, (uint64_t)point_num * (uint64_t)new_dim * sizeof(float));
      #endif

      for(uint64_t i=0; i<point_num; i++){
        memcpy(data_new + i * new_dim, data_ori + i * dim, dim * sizeof(float));
        memset(data_new + i * new_dim + dim, 0, (new_dim - dim) * sizeof(float));
      }
      dim = new_dim;
      #ifdef __APPLE__
        delete[] data_ori;
      #else
        free(data_ori);
      #endif
      return data_new;
    }

}

#endif //EFANNA2E_UTIL_H
