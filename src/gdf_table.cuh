#ifndef GDF_TABLE_H
#define GDF_TABLE_H

#include <gdf/gdf.h>
#include <thrust/device_vector.h>
#include <cassert>
#include "hash-join/hash_functions.cuh"

// TODO Inherit from managed class to allocated with managed memory
class gdf_table 
{

public:

  gdf_table(const size_t num_cols, const gdf_column ** gdf_columns) 
    : num_columns(num_cols), host_columns(gdf_columns)
  {
    // Copy the pointers to the column's data and types to the device 
    // as contiguous arrays
    device_columns.reserve(num_cols);
    device_types.reserve(num_cols);
    for(size_t i = 0; i < num_cols; ++i)
    {
      device_columns.push_back(host_columns[i]->data);
      device_types.push_back(host_columns[i]->dtype);
    }

    d_columns_data = device_columns.data().get();
    d_columns_types = device_types.data().get();
  }

  ~gdf_table(){}

  /* --------------------------------------------------------------------------*/
  /** 
   * @Synopsis  This device function computes a hash value for a given row in the table
   * 
   * @Param row_index The row of the table to compute the hash value for
   * @tparam hash_function The hash function that is used for each element in the row
   * 
   * @Returns The hash value of the row
   */
  /* ----------------------------------------------------------------------------*/
  template <template <typename> class hash_function = default_hash,
            typename T>
  __device__ 
  typename hash_function<T>::result_type hash_row(size_t row_index)
  {
    using hash_value_t = typename hash_function<T>::result_type;
    hash_value_t hash_value{0};

    for(size_t i = 0; i < num_columns; ++i)
    {
      const gdf_dtype current_column_type = d_columns_types[i];

      switch(current_column_type)
      {
        case GDF_INT8:
          {
            using col_type = int8_t;
            hash_function<col_type> hasher;
            const col_type * current_column = static_cast<col_type*>(d_columns_data[i]);
            const col_type current_value = current_column[row_index];
            hash_value_t key_hash = hasher(current_value);
            hash_value = hasher.hash_combine(hash_value, key_hash);
          }
        case GDF_INT16:
          {
            using col_type = int16_t;
            hash_function<col_type> hasher;
            const col_type * current_column = static_cast<col_type*>(d_columns_data[i]);
            const col_type current_value = current_column[row_index];
            hash_value_t key_hash = hasher(current_value);
            hash_value = hasher.hash_combine(hash_value, key_hash);
          }
        case GDF_INT32:
          {
            using col_type = int32_t;
            hash_function<col_type> hasher;
            const col_type * current_column = static_cast<col_type*>(d_columns_data[i]);
            const col_type current_value = current_column[row_index];
            hash_value_t key_hash = hasher(current_value);
            hash_value = hasher.hash_combine(hash_value, key_hash);
          }
        case GDF_INT64:
          {
            using col_type = int64_t;
            hash_function<col_type> hasher;
            const col_type * current_column = static_cast<col_type*>(d_columns_data[i]);
            const col_type current_value = current_column[row_index];
            hash_value_t key_hash = hasher(current_value);
            hash_value = hasher.hash_combine(hash_value, key_hash);
          }
        case GDF_FLOAT32:
          {
            using col_type = float;
            hash_function<col_type> hasher;
            const col_type * current_column = static_cast<col_type*>(d_columns_data[i]);
            const col_type current_value = current_column[row_index];
            hash_value_t key_hash = hasher(current_value);
            hash_value = hasher.hash_combine(hash_value, key_hash);
          }
        case GDF_FLOAT64:
          {
            using col_type = double;
            hash_function<col_type> hasher;
            const col_type * current_column = static_cast<col_type*>(d_columns_data[i]);
            const col_type current_value = current_column[row_index];
            hash_value_t key_hash = hasher(current_value);
            hash_value = hasher.hash_combine(hash_value, key_hash);
          }
        default:
          assert(false && "Attempted to hash unsupported GDF datatype");
      }
    }
  }


private:

  void ** d_columns_data{nullptr};
  gdf_dtype * d_columns_types{nullptr};

  thrust::device_vector<void*> device_columns;
  thrust::device_vector<gdf_dtype> device_types;

  const gdf_column ** host_columns;
  const size_t num_columns;

};

#endif
