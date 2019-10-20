// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package vxform

import (
	"image"

	"github.com/emer/emergent/env"
)

// XForm represents current and previous visual transformation values
// and can apply current values to transform an image.
// Transformations are performed as: rotation, scale, then translation.
// Scaling crops to retain the current image size.
type XForm struct {
	TransX env.CurPrvF32 `desc:"current, prv X-axis (horizontal) translation value, as proportion of image size"`
	TransY env.CurPrvF32 `desc:"current, prv Y-axis (horizontal) translation value, as proportion of image size"`
	Scale  env.CurPrvF32 `desc:"current, prv scale value"`
	Rot    env.CurPrvF32 `desc:"current, prv rotation value, in degrees"`
}

// Set updates current values
func (xf *XForm) Set(trX, trY, sc, rot float32) {
	xf.TransX.Update(trX)
	xf.TransY.Update(trY)
	xf.Scale.Update(sc)
	xf.Rot.Update(rot)
}

// Image transforms given image according to current parameters
func (xf *XForm) Image(img image.Image) *image.RGBA {
	return XFormImage(img, xf.TransX.Cur, xf.TransY.Cur, xf.Scale.Cur, xf.Rot.Cur)
}
