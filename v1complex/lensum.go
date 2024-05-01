// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package v1complex

import (
	"sync"

	"cogentcore.org/core/tensor"
	"github.com/emer/vision/v2/nproc"
)

var (
	// linear neighbor coordinates for 4 angles, also uses negated version
	// -- = (1,0) (X,Y)
	// /  = (1,1)
	// |  = (0,1)
	// \  = (1,-1)
	Line4X = []int{1, 1, 0, 1}
	Line4Y = []int{0, 1, 1, -1}
)

// LenSum4 computes summed line activations.
// If lsum is not same shape as act, it will be
// made so (most efficient to re-use same structure).
// Act must be a 4D tensor with features as inner 2D.
// 4 version ONLY works with 4 angles (inner-most feature dimension)
func LenSum4(act, lsum *tensor.Float32) {
	if !lsum.Shape.IsEqual(&act.Shape) {
		lsum.SetShape(act.Shape.Shp, act.Shape.Strd, act.Shape.Nms)
	}
	plY := act.DimSize(2)
	nang := act.DimSize(3)
	ncpu := nproc.NumCPU()
	nthrs, nper, rmdr := nproc.ThreadNs(ncpu, nang*plY)
	var wg sync.WaitGroup
	for th := 0; th < nthrs; th++ {
		wg.Add(1)
		f := th * nper
		go lenSum4Thr(&wg, f, nper, act, lsum)
	}
	if rmdr > 0 {
		wg.Add(1)
		f := nthrs * nper
		go lenSum4Thr(&wg, f, rmdr, act, lsum)
	}
	wg.Wait()
}

// lenSum4Thr is per-thread implementation
func lenSum4Thr(wg *sync.WaitGroup, fno, nf int, act, lsum *tensor.Float32) {

	acts := act.Values
	lsums := lsum.Values

	layY := act.DimSize(0)
	layX := act.DimSize(1)

	plY := act.DimSize(2)
	nang := act.DimSize(3)
	plN := plY * nang

	norm := float32(1) / 3

	for fi := 0; fi < nf; fi++ {
		ui := fno + fi
		py := ui / nang
		ang := ui % nang
		pi := 0
		for ly := 0; ly < layY; ly++ {
			for lx := 0; lx < layX; lx++ {
				pui := pi * plN
				idx := pui + ui
				ctr := acts[idx]

				lp := float32(0)
				lpX := lx + Line4X[ang]
				lpY := ly + Line4Y[ang]
				if lpX >= 0 && lpX < layX && lpY >= 0 && lpY < layY {
					lp = act.Value([]int{lpY, lpX, py, ang})
				}
				ln := float32(0)
				lnX := lx - Line4X[ang]
				lnY := ly - Line4Y[ang]
				if lnX >= 0 && lnX < layX && lnY >= 0 && lnY < layY {
					ln = act.Value([]int{lnY, lnX, py, ang})
				}
				ls := norm * (ctr + lp + ln)
				lsums[idx] = ls
				pi++
			}
		}
	}
	wg.Done()
}
