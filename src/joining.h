/*
 * Copyright (c) 2018, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/* Header-only join C++ API (high-level) */

#include <limits>
#include <memory>

#include "hash-join/join_compute_api.h"
#include "sort-join.cuh"
#include "gdf_table.cuh"

template<JoinType join_type, 
         typename output_type,
         typename gdf_table_type>
mgpu::mem_t<output_type> join_hash(gdf_table_type const & left_table, 
                                   gdf_table_type const & right_table, 
                                   mgpu::context_t & context) 
{
  mgpu::mem_t<output_type> joined_output;

  const gdf_dtype key_type = left_table.get_build_column_type();

  switch(key_type)
  {
    case GDF_INT8:    
      {
        compute_hash_join<join_type, gdf_table_type, int8_t, output_type>(context, joined_output, left_table, right_table); 
        break;
      }
    case GDF_INT16:   
      {
        compute_hash_join<join_type, gdf_table_type, int16_t, output_type>(context, joined_output, left_table, right_table); 
        break;
      }
    case GDF_INT32:   
      {
        compute_hash_join<join_type, gdf_table_type, int32_t, output_type>(context, joined_output, left_table, right_table); 
        break;
      }
    case GDF_INT64:   
      {
        compute_hash_join<join_type, gdf_table_type, int64_t, output_type>(context, joined_output, left_table, right_table);                    
        break;
      }
    // For floating point types build column, treat as an integral type
    case GDF_FLOAT32: 
      {
        compute_hash_join<join_type, gdf_table_type, int32_t, output_type>(context, joined_output, left_table, right_table);
        break;
      }
    case GDF_FLOAT64: 
      {
        compute_hash_join<join_type, gdf_table_type, int64_t, output_type>(context, joined_output, left_table, right_table);
        break;
      }
    default:
      assert(false && "Invalid build column datatype.");
  }

  return joined_output;
}

struct join_result_base {
  virtual ~join_result_base() {}
  virtual void* data() = 0;
  virtual size_t size() = 0;
};

template <typename T>
struct join_result : public join_result_base {
  mgpu::standard_context_t context;
  mgpu::mem_t<T> result;

  join_result() : context(false) {}
  virtual void* data() {
    return result.data();
  }
  virtual size_t size() {
    return result.size();
  }
};
