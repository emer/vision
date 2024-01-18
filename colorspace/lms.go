// Copyright (c) 2021, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package colorspace

//go:generate core generate

import (
	"cogentcore.org/core/mat32"
)

// LMSComponents are different components of the LMS space
// including opponent contrasts and grey
type LMSComponents int32 //enums:enum

const (
	// Long wavelength = Red component
	LC LMSComponents = iota

	// Medium wavelength = Green component
	MC

	// Short wavelength = Blue component
	SC

	// Long + Medium wavelength = Yellow component
	LMC

	// L - M opponent contrast: Red vs. Green
	LvMC

	// S - L+M opponent contrast: Blue vs. Yellow
	SvLMC

	// achromatic response (grey scale lightness)
	GREY
)

// Opponents enumerates the three primary opponency channels:
// WhiteBlack, RedGreen, BlueYellow
// using colloquial "everyday" terms.
type Opponents int32 //enums:enum

const (
	// White vs. Black greyscale
	WhiteBlack Opponents = iota

	// Red vs. Green
	RedGreen

	// Blue vs. Yellow
	BlueYellow
)

///////////////////////////////////
// CAT02 versions

// XYZToLMS_CAT02 converts XYZ to Long, Medium, Short cone-based responses,
// using the CAT02 transform from CIECAM02 color appearance model
// (MoroneyFairchildHuntEtAl02)
func XYZToLMS_CAT02(x, y, z float32) (l, m, s float32) {
	l = 0.7328*x + 0.4296*y + -0.1624*z
	m = -0.7036*x + 1.6975*y + 0.0061*z
	s = 0.0030*x + 0.0136*y + 0.9834*z
	return
}

// SRGBLinToLMS_CAT02 converts sRGB linear to Long, Medium, Short
// cone-based responses, using the CAT02 transform from CIECAM02
// color appearance model (MoroneyFairchildHuntEtAl02)
// this is good for representing adaptation but NOT apparently
// good for representing appearances
func SRGBLinToLMS_CAT02(rl, gl, bl float32) (l, m, s float32) {
	l = 0.3904054*rl + 0.54994122*gl + 0.00892632*bl
	m = 0.0708416*rl + 0.96317176*gl + 0.00135775*bl
	s = 0.0491304*rl + 0.21556128*gl + 0.9450824*bl
	return
}

// SRGBToLMS_CAT02 converts sRGB to Long, Medium, Short cone-based responses,
// using the CAT02 transform from CIECAM02 color appearance model
// (MoroneyFairchildHuntEtAl02)
func SRGBToLMS_CAT02(r, g, b float32) (l, m, s float32) {
	rl, gl, bl := SRGBToLinear(r, g, b)
	l, m, s = SRGBLinToLMS_CAT02(rl, gl, bl)
	return
}

/*
// #CAT_ColorSpace convert Long, Medium, Short cone-based responses to XYZ, using the CAT02 transform from CIECAM02 color appearance model (MoroneyFairchildHuntEtAl02)
func LMSToXYZ_CAT02(l, m, s float32) (x, y, z float32) {
    x = 1.096124 * l + 0.4296f * Y + -0.1624f * Z;
    y = -0.7036f * X + 1.6975f * Y + 0.0061f * Z;
    z = 0.0030f * X + 0.0136f * Y + 0.9834 * Z;
  }
*/

///////////////////////////////////
// HPE versions

// XYZToLMS_HPE convert XYZ to Long, Medium, Short cone-based responses,
// using the Hunt-Pointer-Estevez transform.
// This is closer to the actual response functions of the L,M,S cones apparently.
func XYZToLMS_HPE(x, y, z float32) (l, m, s float32) {
	l = 0.38971*x + 0.68898*y + -0.07868*z
	m = -0.22981*x + 1.18340*y + 0.04641*z
	s = z
	return
}

// SRGBLinToLMS_HPE converts sRGB linear to Long, Medium, Short cone-based responses,
// using the Hunt-Pointer-Estevez transform.
// This is closer to the actual response functions of the L,M,S cones apparently.
func SRGBLinToLMS_HPE(rl, gl, bl float32) (l, m, s float32) {
	l = 0.30567503*rl + 0.62274014*gl + 0.04530167*bl
	m = 0.15771291*rl + 0.7697197*gl + 0.08807348*bl
	s = 0.0193*rl + 0.1192*gl + 0.9505*bl
	return
}

// SRGBToLMS_HPE converts sRGB to Long, Medium, Short cone-based responses,
// using the Hunt-Pointer-Estevez transform.
// This is closer to the actual response functions of the L,M,S cones apparently.
func SRGBToLMS_HPE(r, g, b float32) (l, m, s float32) {
	rl, gl, bl := SRGBToLinear(r, g, b)
	l, m, s = SRGBLinToLMS_HPE(rl, gl, bl)
	return
}

/*
  func LMStoXYZ_HPE(float& X, float& Y, float& Z,
                                    L, M, S) {
    X = 1.096124f * L + 0.4296f * Y + -0.1624f * Z;
    Y = -0.7036f * X + 1.6975f * Y + 0.0061f * Z;
    Z = 0.0030f * X + 0.0136f * Y + 0.9834 * Z;
  }
  // #CAT_ColorSpace convert Long, Medium, Short cone-based responses to XYZ, using the Hunt-Pointer-Estevez transform -- this is closer to the actual response functions of the L,M,S cones apparently
*/

// LuminanceAdaptation implements the luminance adaptation function
// equals 1 at background luminance of 200 so we generally ignore it..
// bgLum is background luminance -- 200 default.
func LuminanceAdaptation(bgLum float32) float32 {
	lum5 := 5.0 * bgLum
	k := 1.0 / (lum5 + 1)
	k4 := k * k * k * k
	k4m1 := 1 - k4
	fl := 0.2*k4*lum5 + .1*k4m1*k4m1*mat32.Pow(lum5, 1.0/3.0)
	return fl
}

// ResponseCompression takes a 0-1 normalized LMS value
// and performs hyperbolic response compression.
// val must ALREADY have the luminance adaptation applied to it
// using the luminance adaptation function, which is 1 at a
// background luminance level of 200 = 2, so you can skip that
// step if you assume that level of background.
func ResponseCompression(val float32) float32 {
	pval := mat32.Pow(val, 0.42)
	rc := 0.1 + 4.0*pval/(27.13+pval)
	return rc
}

// LMSToComps converts Long, Medium, Short cone-based responses
// to components incl opponents: Red - Green (LvM) and Blue - Yellow (SvLM).
// Includes the separate components in these subtractions as well
// Uses the CIECAM02 color appearance model (MoroneyFairchildHuntEtAl02)
// https://en.wikipedia.org/wiki/CIECAM02
func LMSToComps(l, m, s float32) (lc, mc, sc, lmc, lvm, svlm, grey float32) {
	lrc := ResponseCompression(l)
	mrc := ResponseCompression(m)
	src := ResponseCompression(s)
	// subtract min and mult by 6 gets values roughly into 1-0 range for L,M
	lc = 6.0 * ((lrc + (1.0/11.0)*src) - 0.109091)
	mc = 6.0 * (((12.0 / 11.0) * mrc) - 0.109091)
	lvm = lc - mc // red-green subtracting "criterion for unique yellow"
	lmc = 6.0 * (((1.0 / 9.0) * (lrc + mrc)) - 0.0222222)
	sc = 6.0 * (((2.0 / 9.0) * src) - 0.0222222)
	svlm = sc - lmc // blue-yellow contrast
	grey = (1.0 / 0.431787) * (2.0*lrc + mrc + .05*src - 0.305)
	// note: last term should be: 0.725 * (1/5)^-0.2 = grey background assumption (Yb/Yw = 1/5) = 1
	return
}
