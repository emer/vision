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

// ConvDiff computes difference of two separate filter convolutions
// (fltOn - fltOff) over two images into out.  There are separate gain
// multipliers for On and overall gain.
// images *must* have border (padding) so that filters are
// applied without any bounds checking -- wrapping etc is all
// done in the padding process, which is much more efficient.
// Computation is parallel in image lines.
// img must be a 2D tensor of image values (grey or single components).
// Everything must be organized row major as tensor default.
// Output has 2 outer dims for positive vs. negative values, inner is Y, X
func ConvDiff(geom *Geom, fltOn, fltOff *tensor.Float32, imgOn, imgOff, out *tensor.Float32, gain, gainOn float32) {
	fy := fltOn.DimSize(0)
	fx := fltOn.DimSize(1)

	geom.FiltSz = image.Point{fx, fy}
	geom.UpdtFilt()

	imgSz := image.Point{imgOn.DimSize(1), imgOn.DimSize(0)}
	geom.SetSize(imgSz)
	oshp := []int{2, int(geom.Out.Y), int(geom.Out.X)}
	out.SetShape(oshp, "OnOff", "Y", "X")
	ncpu := nproc.NumCPU()
	nthrs, nper, rmdr := nproc.ThreadNs(ncpu, geom.Out.Y)
	var wg sync.WaitGroup
	for th := 0; th < nthrs; th++ {
		wg.Add(1)
		yst := th * nper
		go convDiffThr(&wg, geom, yst, nper, fltOn, fltOff, imgOn, imgOff, out, gain, gainOn)
	}
	if rmdr > 0 {
		wg.Add(1)
		yst := nthrs * nper
		go convDiffThr(&wg, geom, yst, rmdr, fltOn, fltOff, imgOn, imgOff, out, gain, gainOn)
	}
	wg.Wait()
}

// convDiffThr is per-thread implementation
func convDiffThr(wg *sync.WaitGroup, geom *Geom, yst, ny int, fltOn, fltOff *tensor.Float32, imgOn, imgOff, out *tensor.Float32, gain, gainOn float32) {
	ist := geom.Border.Sub(geom.FiltLt)
	for yi := 0; yi < ny; yi++ {
		y := yst + yi
		iy := int(ist.Y + y*geom.Spacing.Y)
		for x := 0; x < geom.Out.X; x++ {
			ix := ist.X + x*geom.Spacing.X
			var sumOn, sumOff float32
			fi := 0
			for fy := 0; fy < geom.FiltSz.Y; fy++ {
				for fx := 0; fx < geom.FiltSz.X; fx++ {
					idx := imgOn.Shape().Offset([]int{iy + fy, ix + fx})
					sumOn += imgOn.Values[idx] * fltOn.Values[fi]
					sumOff += imgOff.Values[idx] * fltOff.Values[fi]
					fi++
				}
			}
			diff := gain * (gainOn*sumOn - sumOff)
			if diff > 0 {
				out.Set([]int{0, y, x}, diff)
				out.Set([]int{1, y, x}, float32(0))
			} else {
				out.Set([]int{0, y, x}, float32(0))
				out.Set([]int{1, y, x}, -diff)
			}
		}
	}
	wg.Done()
}
