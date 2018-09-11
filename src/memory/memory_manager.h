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

/** ---------------------------------------------------------------------------*
 * @brief Memory Manager class
 * 
 * ---------------------------------------------------------------------------**/

#pragma once

#include <vector>
#include <chrono>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <set>
#include "memory.h"
#include "cnmem.h"

/** ---------------------------------------------------------------------------*
 * @brief Macro wrapper for CNMEM API calls to return appropriate RMM errors.
 * ---------------------------------------------------------------------------**/
#define RMM_CHECK_CNMEM(call) do { \
    cnmemStatus_t error = (call); \
    if (CNMEM_STATUS_CUDA_ERROR == error) \
        return RMM_ERROR_CUDA_ERROR; \
    else if (CNMEM_STATUS_INVALID_ARGUMENT == error) \
        return RMM_ERROR_INVALID_ARGUMENT; \
    else if (CNMEM_STATUS_NOT_INITIALIZED == error) \
        return RMM_ERROR_NOT_INITIALIZED; \
    else if (CNMEM_STATUS_OUT_OF_MEMORY == error) \
        return RMM_ERROR_OUT_OF_MEMORY; \
    else if (CNMEM_STATUS_UNKNOWN_ERROR == error) \
        return RMM_ERROR_UNKNOWN; \
} while(0)

typedef struct CUstream_st *cudaStream_t;

namespace rmm 
{
    // TODO: not currently thread safe
    class Manager
    {
    public:
        Manager() {}

#ifndef RMM_USE_CUDAMALLOC
        /** ---------------------------------------------------------------------------*
         * @brief Register a new stream into the device memory manager.
         * 
         * Also returns success if the stream is already registered.
         * 
         * @param stream The stream to register
         * @return rmmError_t RMM_SUCCESS if all goes well, RMM_ERROR_INVALID_ARGUMENT
         *                    if the stream is invalid.
         * ---------------------------------------------------------------------------**/
        rmmError_t registerStream(cudaStream_t stream) { 
            if (0 == registered_streams.count(stream)) {
                registered_streams.insert(stream);
                if (stream) // don't register the null stream with CNMem
                    RMM_CHECK_CNMEM( cnmemRegisterStream(stream) );
            }
            return RMM_SUCCESS;
        }

    private:
        std::set<cudaStream_t> registered_streams;
#endif
    };

    // TODO: not currently thread safe
    class Logger
    {
    public:        
        Logger() { base_time = std::chrono::system_clock::now(); }

        typedef enum {
            Alloc = 0,
            Realloc,
            Free
        } MemEvent_t;

        using TimePt = std::chrono::system_clock::time_point;

        /// Record a memory manager event in the log.
        void record(MemEvent_t event, int deviceId, void* ptr,
                    TimePt start, TimePt end, 
                    size_t freeMem, size_t totalMem,
                    size_t size=0, cudaStream_t stream=0);
        
        /// Write the log to comma-separated value file
        void to_csv(std::ostream &csv);
    private:
        std::set<void*> current_allocations;

        struct MemoryEvent {
            MemEvent_t event;
            int deviceId;
            void* ptr;
            size_t size;
            cudaStream_t stream;
            size_t freeMem;
            size_t totalMem;
            size_t currentAllocations;
            TimePt start;
            TimePt end;
        };
        
        TimePt base_time;
        std::vector<MemoryEvent> events;
    };
}