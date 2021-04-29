// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package v1complex

import (
	"sync"

	"github.com/emer/etable/etensor"
	"github.com/emer/vision/nproc"
	"github.com/goki/mat32"
)

var (
	// end-stop off coordinates for 4 angles, also uses negated versions
	// these go with the negative versions of Line4X (i.e., are in same dir)
	// -- | = (1,1), (1,0), (1,-1) (X,Y)
	// --|
	// /  = (0,1), (1,1), (1,0)
	// ---
	//  |  = (-1,1), (0,1), (1,1)
	// \   = (0,-1), (1,-1), (1,0)
	// --|
	// 3 coords per angle
	EndStopOff4X = []int{
		1, 1, 1,
		0, 1, 1,
		-1, 0, 1,
		0, 1, 1}
	EndStopOff4Y = []int{
		1, 0, -1,
		1, 1, 0,
		1, 1, 1,
		-1, -1, 0}
)

// EndStop4 computes end-stop activations: es := lsum - max(off)
// lsum is the length-sum activation to the "left" of feature
// and max(off) is the max of the off inhibitory region to the "right"
// of feature.  Both directions are computed, as two rows by angles.
// Act must be a 4D tensor with features as inner 2D.
// 4 version ONLY works with 4 angles (inner-most feature dimension)
func EndStop4(act, lsum, estop *etensor.Float32) {
	layY := act.Dim(0)
	layX := act.Dim(1)

	plY := act.Dim(2)
	nang := act.Dim(3)

	oshp := []int{layY, layX, 2 * plY, nang} // 2 = 2 directions
	if !etensor.EqualInts(oshp, estop.Shp) {
		estop.SetShape(oshp, nil, []string{"Y", "X", "Dir", "Angle"})
	}
	ncpu := nproc.NumCPU()
	nthrs, nper, rmdr := nproc.ThreadNs(ncpu, plY*nang)
	var wg sync.WaitGroup
	for th := 0; th < nthrs; th++ {
		wg.Add(1)
		f := th * nper
		go endStop4Thr(&wg, f, nper, act, lsum, estop)
	}
	if rmdr > 0 {
		wg.Add(1)
		f := nthrs * nper
		go endStop4Thr(&wg, f, rmdr, act, lsum, estop)
	}
	wg.Wait()
}

// endStop4Thr is per-thread implementation
func endStop4Thr(wg *sync.WaitGroup, fno, nf int, act, lsum, estop *etensor.Float32) {
	layY := act.Dim(0)
	layX := act.Dim(1)

	nang := act.Dim(3)

	for fi := 0; fi < nf; fi++ {
		ui := fno + fi
		py := ui / nang
		ang := ui % nang

		for ly := 0; ly < layY; ly++ {
			for lx := 0; lx < layX; lx++ {
				for dir := 0; dir < 2; dir++ {
					dsign := 1
					if dir > 0 {
						dsign = -1
					}
					ls := float32(0)
					// length-sum point is "left" (negative) direction from ctr
					lnX := lx - dsign*Line4X[ang]
					lnY := ly - dsign*Line4Y[ang]
					if lnX >= 0 && lnX < layX && lnY >= 0 && lnY < layY {
						ls = lsum.Value([]int{lnY, lnX, py, ang})
					}

					offMax := float32(0)
					for oi := 0; oi < 3; oi++ {
						ofX := lx + dsign*EndStopOff4X[ang*3+oi]
						ofY := ly + dsign*EndStopOff4Y[ang*3+oi]
						if ofX >= 0 && ofX < layX && ofY >= 0 && ofY < layY {
							off := act.Value([]int{ofY, ofX, py, ang})
							offMax = mat32.Max(offMax, off)
						}
					}
					es := ls - offMax // simple diff
					if es < 0.2 {     // note: builtin threshold
						es = 0
					}
					estop.Set([]int{ly, lx, py*2 + dir, ang}, es)
				}
			}
		}
	}
	wg.Done()
}
