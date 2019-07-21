// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package vfilter

import (
	"image"
	"sync"

	"github.com/emer/etable/etensor"
	"github.com/emer/vision/nproc"
)

// MaxPool performs max-pooling over given pool size and spacing.
// size must = spacing or 2 * spacing.
// Pooling is sensitive to the feature structure of the input, which
// must have shape: Y, X, Polarities, Angles.
func MaxPool(psize, spc image.Point, in, out *etensor.Float32) {
	ny := in.Dim(0)
	nx := in.Dim(1)
	pol := in.Dim(2)
	nang := in.Dim(3)
	oy := ny / int(spc.Y)
	ox := nx / int(spc.X)
	if spc.Y != psize.Y {
		oy--
	}
	if spc.X != psize.X {
		ox--
	}

	oshp := []int{oy, ox, pol, nang}
	if !etensor.EqualInts(oshp, out.Shp) {
		out.SetShape(oshp, nil, []string{"Y", "X", "Polarity", "Angle"})
	}
	nf := pol * nang
	ncpu := nproc.NumCPU()
	nthrs, nper, rmdr := nproc.ThreadNs(ncpu, nf)
	var wg sync.WaitGroup
	for th := 0; th < nthrs; th++ {
		wg.Add(1)
		f := th * nper
		go MaxPoolOne(&wg, f, nper, psize, spc, in, out)
	}
	if rmdr > 0 {
		wg.Add(1)
		f := nthrs * nper
		go MaxPoolOne(&wg, f, nper, psize, spc, in, out)
	}
	wg.Wait()
}

// MaxPoolOne performs max pooling for one specific polarity, angle
// called as a goroutine from MaxPool
func MaxPoolOne(wg *sync.WaitGroup, fno, nf int, psize, spc image.Point, in, out *etensor.Float32) {
	ny := out.Dim(0)
	nx := out.Dim(1)
	nang := out.Dim(3)
	for fi := 0; fi < nf; fi++ {
		f := fno + fi
		pol := f / nang
		ang := f % nang
		for y := 0; y < ny; y++ {
			iy := y * int(spc.Y)
			for x := 0; x < nx; x++ {
				ix := x * int(spc.X)
				max := float32(0)
				for py := 0; py < int(psize.Y); py++ {
					for px := 0; px < int(psize.X); px++ {
						iv := in.Value([]int{iy + py, ix + px, pol, ang})
						if iv > max {
							max = iv
						}
					}
				}
				out.Set([]int{y, x, pol, ang}, max)
			}
		}
	}
	wg.Done()
}
