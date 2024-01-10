// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

//go:generate goki generate

import (
	"image"
	"log"

	"github.com/anthonynsimon/bild/transform"
	"github.com/emer/vision/v2/dog"
	"github.com/emer/vision/v2/vfilter"
	"goki.dev/etable/v2/etable"
	"goki.dev/etable/v2/etensor"
	_ "goki.dev/etable/v2/etview" // include to get gui views
	"goki.dev/gi/v2/gi"
	"goki.dev/gi/v2/giv"
	"goki.dev/grows/images"
)

func main() {
	vi := &Vis{}
	vi.Defaults()
	vi.Filter()
	vi.ConfigGUI()
}

// Vis encapsulates specific visual processing pipeline in
// use in a given case -- can add / modify this as needed
type Vis struct { //gti:add

	// name of image file to operate on
	ImageFile gi.FileName

	// LGN DoG filter parameters
	DoG dog.Filter

	// geometry of input, output
	Geom vfilter.Geom `edit:"-"`

	// target image size to use -- images will be rescaled to this size
	ImgSize image.Point

	// DoG filter tensor -- has 3 filters (on, off, net)
	DoGTsr etensor.Float32 `view:"no-inline"`

	// DoG filter table (view only)
	DoGTab etable.Table `view:"no-inline"`

	// current input image
	Img image.Image `view:"-"`

	// input image as tensor
	ImgTsr etensor.Float32 `view:"no-inline"`

	// DoG filter output tensor
	OutTsr etensor.Float32 `view:"no-inline"`
}

func (vi *Vis) Defaults() {
	vi.ImageFile = gi.FileName("side-tee-128.png")
	vi.DoG.Defaults()
	sz := 12 // V1mF16 typically = 12, no border
	spc := 4
	vi.DoG.SetSize(sz, spc)
	// note: first arg is border -- we are relying on Geom
	// to set border to .5 * filter size
	// any further border sizes on same image need to add Geom.FiltRt!
	vi.Geom.Set(image.Point{0, 0}, image.Point{spc, spc}, image.Point{sz, sz})
	vi.ImgSize = image.Point{128, 128}
	// vi.ImgSize = image.Point{64, 64}
	vi.DoG.ToTensor(&vi.DoGTsr)
	vi.DoG.ToTable(&vi.DoGTab) // note: view only, testing
	vi.DoGTab.Cols[1].SetMetaData("max", "0.2")
	vi.DoGTab.Cols[1].SetMetaData("min", "-0.2")
}

// OpenImage opens given filename as current image Img
// and converts to a float32 tensor for processing
func (vi *Vis) OpenImage(filepath string) error { //gti:add
	var err error
	vi.Img, _, err = images.Open(filepath)
	if err != nil {
		log.Println(err)
		return err
	}
	isz := vi.Img.Bounds().Size()
	if isz != vi.ImgSize {
		vi.Img = transform.Resize(vi.Img, vi.ImgSize.X, vi.ImgSize.Y, transform.Linear)
	}
	vfilter.RGBToGrey(vi.Img, &vi.ImgTsr, vi.Geom.FiltRt.X, false) // pad for filt, bot zero
	vfilter.WrapPad(&vi.ImgTsr, vi.Geom.FiltRt.X)
	vi.ImgTsr.SetMetaData("image", "+")
	return nil
}

// LGNDoG runs DoG filtering on input image
// must have valid Img in place to start.
func (vi *Vis) LGNDoG() {
	flt := vi.DoG.FilterTensor(&vi.DoGTsr, dog.Net)
	vfilter.Conv1(&vi.Geom, flt, &vi.ImgTsr, &vi.OutTsr, vi.DoG.Gain)
	// log norm is generally good it seems for dogs
	vfilter.TensorLogNorm32(&vi.OutTsr, 0) // 0 = renorm all, 1 = renorm within each on / off separately
}

// Filter is overall method to run filters on current image file name
// loads the image from ImageFile and then runs filters
func (vi *Vis) Filter() error { //gti:add
	err := vi.OpenImage(string(vi.ImageFile))
	if err != nil {
		log.Println(err)
		return err
	}
	vi.LGNDoG()
	return nil
}

//////////////////////////////////////////////////////////////////////////////
// 		Gui

func (vi *Vis) ConfigGUI() *gi.Body {
	b := gi.NewAppBody("lgn_dog").SetTitle("LGN DoG Filtering")
	b.App().About = `This demonstrates LGN-level Difference of Gaussian Filtering.  See <a href="https://github.com/emer/vision/v1">V1 on GitHub</a>.</p>`

	giv.NewStructView(b, "sv").SetStruct(vi)

	b.AddAppBar(func(tb *gi.Toolbar) {
		giv.NewFuncButton(tb, vi.Filter)
		// gi.NewSeparator(tb)
		// vi.Img.ConfigToolbar(tb)
		// gi.NewSeparator(tb)
		// vi.OutAll.ConfigToolbar(tb)
	})

	b.NewWindow().Run().Wait()
	return b
}
