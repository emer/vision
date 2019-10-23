// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package vxform

import (
	"fmt"
	"image"

	"github.com/emer/emergent/env"
)

// XForm represents current and previous visual transformation values
// and can apply current values to transform an image.
// Transformations are performed as: rotation, scale, then translation.
// Scaling crops to retain the current image size.
type XForm struct {
	TransX env.CurPrvF32 `desc:"current, prv X-axis (horizontal) translation value, as proportion of image half-size (i.e., 1 = move from center to edge)"`
	TransY env.CurPrvF32 `desc:"current, prv Y-axis (horizontal) translation value, as proportion of image half-size (i.e., 1 = move from center to edge)"`
	Scale  env.CurPrvF32 `desc:"current, prv scale value"`
	Rot    env.CurPrvF32 `desc:"current, prv rotation value, in degrees"`
}

// Set updates current values
func (xf *XForm) Set(trX, trY, sc, rot float32) {
	xf.TransX.Set(trX)
	xf.TransY.Set(trY)
	xf.Scale.Set(sc)
	xf.Rot.Set(rot)
}

// Image transforms given image according to current parameters
func (xf *XForm) Image(img image.Image) *image.RGBA {
	return XFormImage(img, xf.TransX.Cur, xf.TransY.Cur, xf.Scale.Cur, xf.Rot.Cur)
}

func (xf *XForm) String() string {
	return fmt.Sprintf("tX: %.4f, tY: %.4f, Sc: %.4f, Rt: %.4f", xf.TransX.Cur, xf.TransY.Cur, xf.Scale.Cur, xf.Rot.Cur)
}
