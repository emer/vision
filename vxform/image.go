// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package vxform

import (
	"image"
	"math"

	"github.com/anthonynsimon/bild/clone"
	"github.com/anthonynsimon/bild/transform"
)

// XFormImage transforms given image according to given parameters
// Transformations are performed as: rotation, scale, then translation.
// Scaling retain the current image size, filling border with current border
// if scaling to a smaller size.
func XFormImage(img image.Image, trX, trY, sc, rot float32) *image.RGBA {
	cimg := img
	if rot != 0 {
		cimg = RotImage(cimg, rot)
	}
	if sc != 1 && sc > 0 {
		cimg = ScaleImage(cimg, sc)
	}
	if trX != 0 || trY != 0 {
		cimg = TransImage(cimg, trX, trY)
	}
	return cimg.(*image.RGBA)
}

// RotImage rotates image by given number of degrees
func RotImage(img image.Image, rot float32) *image.RGBA {
	return transform.Rotate(img, float64(rot), nil) // default options: center, crop
}

// ScaleImage scales image by given number of degrees
// retaining the current image size, and filling border with current border
// if scaling to a smaller size.
func ScaleImage(img image.Image, sc float32) *image.RGBA {
	sz := img.Bounds().Size()
	nsz := sz
	nsz.X = int(math.Round(float64(nsz.X) * float64(sc)))
	nsz.Y = int(math.Round(float64(nsz.Y) * float64(sc)))
	simg := transform.Resize(img, nsz.X, nsz.Y, transform.Linear)
	if sc < 1 {
		psz := sz.Sub(nsz).Div(2)
		simg = clone.Pad(simg, psz.X, psz.Y, clone.EdgeExtend)
		rsz := nsz.Add(psz).Add(psz)
		if rsz != sz {
			simg = transform.Crop(simg, image.Rectangle{Max: sz})
		}
	}
	return simg
}

// TransImage translates image in each axis by given proportion of image half-size
// i.e., 1 = move from center to edge
func TransImage(img image.Image, trX, trY float32) *image.RGBA {
	sz := img.Bounds().Size()
	off := sz
	off.X = int(math.Round(0.5 * float64(off.X) * float64(trX)))
	off.Y = int(math.Round(0.5 * float64(off.Y) * float64(trY)))
	return transform.Translate(img, off.X, off.Y)
}
