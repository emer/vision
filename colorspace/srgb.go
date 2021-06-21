// Copyright (c) 2021, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package colorspace

import "github.com/goki/mat32"

// SRGBToLinearComp converts an sRGB rgb component to linear space (removes gamma).
// Used in converting from sRGB to XYZ colors.
func SRGBToLinearComp(srgb float32) float32 {
	if srgb <= 0.04045 {
		return srgb / 12.92
	}
	return mat32.Pow((srgb+0.055)/1.055, 2.4)
}

// SRGBFmLinearComp converts an sRGB rgb linear component
// to non-linear (gamma corrected) sRGB value
// Used in converting from XYZ to sRGB.
func SRGBFmLinearComp(lin float32) float32 {
	if lin <= 0.0031308 {
		return 12.92 * lin
	}
	return (1.055*mat32.Pow(lin, 1/2.4) + 0.055)
}

// SRGBToLinear converts set of sRGB components to linear values,
// removing gamma correction.
func SRGBToLinear(r, g, b float32) (rl, gl, bl float32) {
	rl = SRGBToLinearComp(r)
	gl = SRGBToLinearComp(g)
	bl = SRGBToLinearComp(b)
	return
}

// SRGBFmLinear converts set of sRGB components from linear values,
// adding gamma correction.
func SRGBFmLinear(rl, gl, bl float32) (r, g, b float32) {
	r = SRGBFmLinearComp(rl)
	g = SRGBFmLinearComp(gl)
	b = SRGBFmLinearComp(bl)
	return
}

// SRGBToOpponents converts sRGB to opponent components via LMS
// using the HPE cone values: Red - Green (LvM) and Blue - Yellow (SvLM).
// Includes the separate components in these subtractions as well.
// Uses the CIECAM02 color appearance model (MoroneyFairchildHuntEtAl02)
// https://en.wikipedia.org/wiki/CIECAM02
// using the Hunt-Pointer-Estevez transform.
func SRGBToOpponents(r, g, b float32) (lc, mc, sc, lmc, lvm, svlm, grey float32) {
	l, m, s := SRGBToLMS_HPE(r, g, b) // note: HPE
	lc, mc, sc, lmc, lvm, svlm, grey = LMSToOpponents(l, m, s)
	return
}

/*
 func SRGBtoOpponents_GenLookup();
  // ensure that the lookup table for srgb to opponents is built -- returns true if needed to build

  static void sRGBtoOpponents_lkup(float& L_c, float& M_c, float& S_c, float& LM_c,
                                   float& LvM, float& SvLM, float& grey,
                                   r_s, g_s, b_s);
  // #CAT_ColorSpace Lookup table version: convert sRGB to opponent components via LMS using the HPE cone values: Red - Green (LvM) and Blue - Yellow (SvLM) -- includes the separate components in these subtractions as well -- uses the CIECAM02 color appearance model (MoroneyFairchildHuntEtAl02) https://en.wikipedia.org/wiki/CIECAM02
*/
