#include <cstdlib>
#include <iostream>
#include <vector>

#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/gather.h>

#include "gtest/gtest.h"
#include "gmock/gmock.h"
#include <gdf/gdf.h>
#include <gdf/cffi/functions.h>

using namespace testing;

gdf_column
create_gdf_column(thrust::device_vector<int> &d) {
    gdf_column c = {thrust::raw_pointer_cast(d.data()), nullptr, d.size(), GDF_INT32, TIME_UNIT_NONE};
    return c;
}

std::vector<int> host_vec(thrust::device_vector<int> &dev_vec) {
    std::vector<int> data(dev_vec.size());
    thrust::copy(dev_vec.begin(), dev_vec.end(), data.begin());
    return data;
}

gdf_error
call_gdf_test(
        std::array<thrust::device_vector<int>, 3> &l,
        std::array<thrust::device_vector<int>, 3> &r,
        thrust::device_vector<int> &out_left_pos,
        thrust::device_vector<int> &out_right_pos,
        const int index) {
    std::vector<int> l0{0, 0, 4, 5, 5};
    std::vector<int> l1{1, 2, 2, 3, 4};
    std::vector<int> l2{1, 1, 3, 1, 2};
    std::vector<int> r0{0, 0, 2, 3, 5};
    std::vector<int> r1{1, 2, 3, 3, 4};
    std::vector<int> r2{3, 3, 2, 1, 1};

    thrust::device_vector<int> dl0 = l0; thrust::swap(dl0, l[0]);
    thrust::device_vector<int> dl1 = l1; thrust::swap(dl1, l[1]);
    thrust::device_vector<int> dl2 = l2; thrust::swap(dl2, l[2]);
    thrust::device_vector<int> dr0 = r0; thrust::swap(dr0, r[0]);
    thrust::device_vector<int> dr1 = r1; thrust::swap(dr1, r[1]);
    thrust::device_vector<int> dr2 = r2; thrust::swap(dr2, r[2]);

    gdf_column gdl0 = create_gdf_column(l[0]);
    gdf_column gdl1 = create_gdf_column(l[1]);
    gdf_column gdl2 = create_gdf_column(l[2]);

    gdf_column gdr0 = create_gdf_column(r[0]);
    gdf_column gdr1 = create_gdf_column(r[1]);
    gdf_column gdr2 = create_gdf_column(r[2]);

    gdf_column* gl[3] = {&gdl0, &gdl1, &gdl2};
    gdf_column* gr[3] = {&gdr0, &gdr1, &gdr2};
    gdf_join_result_type *out;
    gdf_error err = gdf_multi_left_join_generic(index, gl, gr, &out);

    size_t len = gdf_join_result_size(out);
    size_t hlen = len/2;
    int* out_ptr = reinterpret_cast<int*>(gdf_join_result_data(out));
    thrust::device_vector<int> out_data(out_ptr, out_ptr + len);

    thrust::sort_by_key(out_data.begin() + hlen, out_data.end(), out_data.begin());
    thrust::sort_by_key(out_data.begin(), out_data.begin() + hlen, out_data.begin() + hlen);
    out_left_pos.resize(hlen);
    out_right_pos.resize(hlen);

    thrust::copy(out_data.begin(), out_data.begin() + out_left_pos.size(), out_left_pos.begin());
    thrust::copy(out_data.begin() + out_right_pos.size(), out_data.end(), out_right_pos.begin());
    return err;
}

TEST(gdf_multi_left_join_TEST, case1) {
    std::array<thrust::device_vector<int>, 3> l;
    std::array<thrust::device_vector<int>, 3> r;
    thrust::device_vector<int> l_pos;
    thrust::device_vector<int> r_pos;
    auto err = call_gdf_test(l, r, l_pos, r_pos, 1);
    thrust::device_vector<int> map_out(l_pos.size());

    EXPECT_THAT(host_vec(l_pos), ElementsAre(0, 0, 1, 1, 2, 3, 4));
    EXPECT_THAT(host_vec(r_pos), ElementsAre(0, 1, 0, 1, -1, 4, 4));

    thrust::gather(l_pos.begin(), l_pos.end(), l[0].begin(), map_out.begin());
    EXPECT_THAT(host_vec(map_out), ElementsAre(0, 0, 0, 0, 4, 5, 5));

    ASSERT_EQ(err, GDF_SUCCESS);
}

TEST(gdf_multi_left_join_TEST, case2) {
    std::array<thrust::device_vector<int>, 3> l;
    std::array<thrust::device_vector<int>, 3> r;
    thrust::device_vector<int> l_pos;
    thrust::device_vector<int> r_pos;
    auto err = call_gdf_test(l, r, l_pos, r_pos, 2);
    thrust::device_vector<int> map_out(l_pos.size());

    EXPECT_THAT(host_vec(l_pos), ElementsAre(0, 1, 2, 3, 4));

    {
        thrust::gather(l_pos.begin(), l_pos.end(), l[0].begin(), map_out.begin());
        EXPECT_THAT(host_vec(map_out), ElementsAre(0, 0, 4, 5, 5));
    }

    {
        thrust::gather(l_pos.begin(), l_pos.end(), l[1].begin(), map_out.begin());
        EXPECT_THAT(host_vec(map_out), ElementsAre(1, 2, 2, 3, 4));
    }

    ASSERT_EQ(err, GDF_SUCCESS);
}

TEST(gdf_multi_left_join_TEST, case3) {
    std::array<thrust::device_vector<int>, 3> l;
    std::array<thrust::device_vector<int>, 3> r;
    thrust::device_vector<int> l_pos;
    thrust::device_vector<int> r_pos;
    auto err = call_gdf_test(l, r, l_pos, r_pos, 2);
    thrust::device_vector<int> map_out(l_pos.size());

    EXPECT_THAT(host_vec(l_pos), ElementsAre(0, 1, 2, 3, 4));

    {
        thrust::gather(l_pos.begin(), l_pos.end(), l[0].begin(), map_out.begin());
        EXPECT_THAT(host_vec(map_out), ElementsAre(0, 0, 4, 5, 5));
    }

    {
        thrust::gather(l_pos.begin(), l_pos.end(), l[1].begin(), map_out.begin());
        EXPECT_THAT(host_vec(map_out), ElementsAre(1, 2, 2, 3, 4));
    }

    {
        thrust::gather(l_pos.begin(), l_pos.end(), l[2].begin(), map_out.begin());
        EXPECT_THAT(host_vec(map_out), ElementsAre(1, 1, 3, 1, 2));
    }

    ASSERT_EQ(err, GDF_SUCCESS);
}