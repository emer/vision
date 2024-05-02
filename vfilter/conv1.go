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

// Conv1 performs convolution of single filter over img into out.
// img *must* have border (padding) so that filters are
// applied without any bounds checking -- wrapping etc is all
// done in the padding process, which is much more efficient.
// Computation is parallel in image lines.
// img must be a 2D tensor of image values (convert RGB to grey first).
// Everything must be organized row major as tensor default.
// Output has 2 outer dims for positive vs. negative values, inner is Y, X
// todo: add option to interleave polarity as inner-most dim.
func Conv1(geom *Geom, flt *tensor.Float32, img, out *tensor.Float32, gain float32) {
	fy := flt.DimSize(0)
	fx := flt.DimSize(1)

	geom.FiltSz = image.Point{fx, fy}
	geom.UpdtFilt()

	imgSz := image.Point{img.DimSize(1), img.DimSize(0)}
	geom.SetSize(imgSz)
	oshp := []int{2, int(geom.Out.Y), int(geom.Out.X)}
	out.SetShape(oshp, "OnOff", "Y", "X")
	ncpu := nproc.NumCPU()
	nthrs, nper, rmdr := nproc.ThreadNs(ncpu, geom.Out.Y)
	var wg sync.WaitGroup
	for th := 0; th < nthrs; th++ {
		wg.Add(1)
		yst := th * nper
		go conv1Thr(&wg, geom, yst, nper, flt, img, out, gain)
	}
	if rmdr > 0 {
		wg.Add(1)
		yst := nthrs * nper
		go conv1Thr(&wg, geom, yst, rmdr, flt, img, out, gain)
	}
	wg.Wait()
}

// conv1Thr is per-thread implementation
func conv1Thr(wg *sync.WaitGroup, geom *Geom, yst, ny int, flt *tensor.Float32, img, out *tensor.Float32, gain float32) {
	ist := geom.Border.Sub(geom.FiltLt)
	for yi := 0; yi < ny; yi++ {
		y := yst + yi
		iy := int(ist.Y + y*geom.Spacing.Y)
		for x := 0; x < geom.Out.X; x++ {
			ix := ist.X + x*geom.Spacing.X
			sum := float32(0)
			fi := 0
			for fy := 0; fy < geom.FiltSz.Y; fy++ {
				for fx := 0; fx < geom.FiltSz.X; fx++ {
					iv := img.Value([]int{iy + fy, ix + fx})
					fv := flt.Values[fi]
					sum += iv * fv
					fi++
				}
			}
			sum *= gain
			if sum > 0 {
				out.Set([]int{0, y, x}, sum)
				out.Set([]int{1, y, x}, float32(0))
			} else {
				out.Set([]int{0, y, x}, float32(0))
				out.Set([]int{1, y, x}, -sum)
			}
		}
	}
	wg.Done()
}
