// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package vfilter

import (
	"image"
	"sync"

	"cogentcore.org/core/tensor"
	"github.com/emer/vision/v2/nproc"
)

// MaxPool performs max-pooling over given pool size and spacing.
// size must = spacing or 2 * spacing.
// Pooling is sensitive to the feature structure of the input, which
// must have shape: Y, X, Polarities, Angles.
func MaxPool(psize, spc image.Point, in, out *tensor.Float32) {
	ny := in.DimSize(0)
	nx := in.DimSize(1)
	pol := in.DimSize(2)
	nang := in.DimSize(3)
	oy := ny / int(spc.Y)
	ox := nx / int(spc.X)
	if spc.Y != psize.Y {
		oy--
	}
	if spc.X != psize.X {
		ox--
	}

	oshp := []int{oy, ox, pol, nang}
	if !tensor.EqualInts(oshp, out.Shp) {
		out.SetShape(oshp, nil, []string{"Y", "X", "Polarity", "Angle"})
	}
	nf := pol * nang
	ncpu := nproc.NumCPU()
	nthrs, nper, rmdr := nproc.ThreadNs(ncpu, nf)
	var wg sync.WaitGroup
	for th := 0; th < nthrs; th++ {
		wg.Add(1)
		f := th * nper
		go maxPoolThr(&wg, f, nper, psize, spc, in, out)
	}
	if rmdr > 0 {
		wg.Add(1)
		f := nthrs * nper
		go maxPoolThr(&wg, f, rmdr, psize, spc, in, out)
	}
	wg.Wait()
}

// maxPoolThr is per-thread implementation
func maxPoolThr(wg *sync.WaitGroup, fno, nf int, psize, spc image.Point, in, out *tensor.Float32) {
	ny := out.DimSize(0)
	nx := out.DimSize(1)
	nang := out.DimSize(3)
	for fi := 0; fi < nf; fi++ {
		f := fno + fi
		pol := f / nang
		ang := f % nang
		for y := 0; y < ny; y++ {
			iy := y * spc.Y
			for x := 0; x < nx; x++ {
				ix := x * spc.X
				max := float32(0)
				for py := 0; py < psize.Y; py++ {
					for px := 0; px < psize.X; px++ {
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
