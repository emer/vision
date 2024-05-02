// Copyright (c) 2021, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package colorspace

import (
	"image"

	"cogentcore.org/core/tensor"
	"github.com/emer/vision/v2/vfilter"
)

// RGBImgLMSComps converts an RGB image to corresponding LMS components
// including color opponents, with components as the outer-most dimension.
// padWidth is the amount of padding to add on all sides.
// topZero retains the Y=0 value at the top of the tensor --
// otherwise it is flipped with Y=0 at the bottom to be consistent
// with the emergent / OpenGL standard coordinate system
func RGBImgToLMSComps(img image.Image, tsr *tensor.Float32, padWidth int, topZero bool) {
	rgbtsr := &tensor.Float32{}
	vfilter.RGBToTensor(img, rgbtsr, padWidth, topZero)
	RGBTensorToLMSComps(rgbtsr, tsr)
}

// RGBTensorToLMSComps converts an RGB Tensor to corresponding LMS components
// including color opponents, with components as the outer-most dimension,
// and assumes rgb is 3 dimensional with outer-most dimension as RGB.
func RGBTensorToLMSComps(tsr *tensor.Float32, rgb *tensor.Float32) {
	sy := rgb.DimSize(1)
	sx := rgb.DimSize(2)
	tsr.SetShape([]int{int(LMSComponentsN), sy, sx}, "LMS", "Y", "X")
	for y := 0; y < sy; y++ {
		for x := 0; x < sx; x++ {
			r := rgb.Value([]int{0, y, x})
			g := rgb.Value([]int{1, y, x})
			b := rgb.Value([]int{2, y, x})
			lc, mc, sc, lmc, lvm, svlm, grey := SRGBToLMSComps(r, g, b)
			tsr.Set([]int{int(LC), y, x}, lc)
			tsr.Set([]int{int(MC), y, x}, mc)
			tsr.Set([]int{int(SC), y, x}, sc)
			tsr.Set([]int{int(LMC), y, x}, lmc)
			tsr.Set([]int{int(LvMC), y, x}, lvm)
			tsr.Set([]int{int(SvLMC), y, x}, svlm)
			tsr.Set([]int{int(GREY), y, x}, grey)
		}
	}
}
