#ifndef GDF_TABLE_H
#define GDF_TABLE_H

#include <gdf/gdf.h>
#include <thrust/device_vector.h>

// TODO Inherit from managed class to allocated with managed memory
class gdf_table 
{

  gdf_table(size_t num_cols, gdf_column * gdf_columns[]) : num_columns{num_cols}, the_columns{gdf_columns}
  {

    for(size_t i = 0; i < num_cols; ++i)
    {
      device_columns.push_back(*gdf_columns[i]);
    }

  }

  thrust::device_vector<gdf_column> device_columns;

  gdf_column * the_columns[];

  const size_t num_columns;

}

#endif
