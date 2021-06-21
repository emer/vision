// Copyright (c) 2021, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package colorspace

import (
	"fmt"
	"math/rand"
	"testing"

	"github.com/goki/mat32"
)

func TestSRGBTable(t *testing.T) {
	tol := float32(1.0e-4)
	for i := 0; i < 10; i++ {
		r := rand.Float32()
		g := rand.Float32()
		b := rand.Float32()
		lc, mc, sc, lmc, lvm, svlm, grey := SRGBToOpponents(r, g, b)
		lcl, mcl, scl, lmcl, lvml, svlml, greyl := TheSRGBToOp.Lookup(r, g, b)
		if mat32.Abs(lc-lcl) > tol {
			fmt.Printf("lc err: comp: %g  lookup: %g\n", lc, lcl)
		}
		if mat32.Abs(mc-mcl) > tol {
			fmt.Printf("mc err: comp: %g  lookup: %g\n", mc, mcl)
		}
		if mat32.Abs(sc-scl) > tol {
			fmt.Printf("sc err: comp: %g  lookup: %g\n", sc, scl)
		}
		if mat32.Abs(lmc-lmcl) > tol {
			fmt.Printf("lmc err: comp: %g  lookup: %g\n", lmc, lmcl)
		}
		if mat32.Abs(lvm-lvml) > tol {
			fmt.Printf("lvm err: comp: %g  lookup: %g\n", lvm, lvml)
		}
		if mat32.Abs(svlm-svlml) > tol {
			fmt.Printf("svlm err: comp: %g  lookup: %g\n", svlm, svlml)
		}
		if mat32.Abs(grey-greyl) > tol {
			fmt.Printf("grey err: comp: %g  lookup: %g\n", grey, greyl)
		}
	}
}
