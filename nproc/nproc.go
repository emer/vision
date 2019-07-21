// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
Package nproc provides number of processors using slurm env var
SLURM_CPUS_PER_TASK or runtime.NumCPU().

TODO: move this to dmem package once that is started.
*/
package nproc

import (
	"os"
	"runtime"
	"strconv"
)

var NumCPUCache int

func NumCPU() int {
	if NumCPUCache > 0 {
		return NumCPUCache
	}
	ncs, ok := os.LookupEnv("SLURM_CPUS_PER_TASK")
	if !ok {
		NumCPUCache = runtime.NumCPU()
	} else {
		NumCPUCache, _ = strconv.Atoi(ncs)
	}
	return NumCPUCache
}

// ThreadNs computes number of threads and number of jobs per thread,
// based on number of cpu's and total number of jobs.
// rmdr is remainder of jobs not evenly divisible by ncpu
func ThreadNs(ncpu, njobs int) (nthrs, nper, rmdr int) {
	if njobs <= ncpu {
		return njobs, 1, 0
	}
	nthrs = ncpu
	nper = njobs / ncpu
	rmdr = njobs % ncpu
	return
}
