// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package v1complex

import (
	"github.com/emer/etable/etensor"
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
func LenSum4(act, lsum *etensor.Float32) {
	if !lsum.Shape.IsEqual(&act.Shape) {
		lsum.SetShape(act.Shape.Shp, act.Shape.Strd, act.Shape.Nms)
	}
	acts := act.Values
	lsums := lsum.Values

	layY := act.Dim(0)
	layX := act.Dim(1)

	plY := act.Dim(2)
	plX := act.Dim(3)
	plN := plY * plX

	norm := float32(1) / 3

	pi := 0
	for ly := 0; ly < layY; ly++ {
		for lx := 0; lx < layX; lx++ {
			pui := pi * plN
			ui := 0
			for py := 0; py < plY; py++ {
				for ang := 0; ang < plX; ang++ {
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
					ui++
				}
			}
			pi++
		}
	}
}
