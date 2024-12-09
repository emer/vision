// Copyright (c) 2021, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package colorspace

import (
	"cogentcore.org/core/math32/vecint"
	"cogentcore.org/core/tensor"
)

// MacbethImage sets the Macbeth standard color test image to given tensor
// with given size and border width around edges.
// if img == nil it is created, and size enforced.
func MacbethImage(img *tensor.Float32, width, height, bord int) {
	sRGBvals := []int{115, 82, 68, // 'Dark Skin';
		194, 150, 130, // 'Light Skin';
		98, 122, 157, // 'Blue Sky';
		87, 108, 67, // 'Foliage';
		133, 128, 177, // 'Blue Flower';
		103, 189, 170, // 'Bluish Green';
		214, 126, 44, // 'Orange';
		80, 91, 166, // 'Purple Red';
		193, 90, 99, // 'Moderate Red';
		94, 60, 108, // 'Purple';
		157, 188, 64, // 'Yellow Green';
		224, 163, 46, // 'Orange Yellow';
		56, 61, 150, // 'Blue';
		70, 148, 73, // 'Green';
		175, 54, 60, // 'Red';
		231, 199, 31, // 'Yellow';
		187, 86, 149, // 'Magenta';
		8, 133, 161, // 'Cyan';
		255, 255, 255, // 'White';
		200, 200, 200, // 'Neutral 8';
		160, 160, 160, // 'Neutral 65';
		122, 122, 121, // 'Neutral 5';
		85, 85, 85, // 'Neutral 35';
		52, 52, 52,
	}

	nsq := vecint.Vector2i{6, 4}
	numsq := nsq.X * nsq.Y
	sz := vecint.Vector2i{width + bord*2 + 8, height + bord*2 + 8}
	bvec := vecint.Vector2i{bord, bord}
	marg := vecint.Vector2i{8, 8}
	upBord := sz.Sub(bvec).Sub(marg)

	netSz := vecint.Vector2i{width, height}
	sqSz := netSz.Div(nsq)

	if img == nil {
		img = &tensor.Float32{}
	}
	img.SetShapeSizes(3, sz.Y, sz.X)

	ic := vecint.Vector2i{}
	for ic.Y = bvec.Y; ic.Y < upBord.Y; ic.Y++ {
		for ic.X = bvec.X; ic.X < upBord.X; ic.X++ {
			nc := ic.Sub(bvec)
			sqc := nc.Div(sqSz)
			sqst := sqc.Mul(sqSz).Add(bvec)
			ps := ic.Sub(sqst)
			if ps.X > marg.X && ps.Y > marg.Y {
				clri := (nsq.Y-1-sqc.Y)*nsq.X + sqc.X
				if clri < numsq {
					r := float32(sRGBvals[clri*3]) / 255
					g := float32(sRGBvals[clri*3+1]) / 255
					b := float32(sRGBvals[clri*3+2]) / 255

					img.Set(r, 0, ic.Y, ic.X)
					img.Set(g, 1, ic.Y, ic.X)
					img.Set(b, 2, ic.Y, ic.X)
				}
			}
		}
	}
}
