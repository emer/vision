// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

//go:generate core generate -add-types

import (
	"image"
	"log"

	"cogentcore.org/core/gi"
	"cogentcore.org/core/giv"
	"cogentcore.org/core/grows/images"
	"github.com/anthonynsimon/bild/transform"
	"github.com/emer/etable/v2/etable"
	"github.com/emer/etable/v2/etensor"
	_ "github.com/emer/etable/v2/etview" // include to get gui views
	"github.com/emer/etable/v2/norm"
	"github.com/emer/vision/v2/fffb"
	"github.com/emer/vision/v2/gabor"
	"github.com/emer/vision/v2/kwta"
	"github.com/emer/vision/v2/v1complex"
	"github.com/emer/vision/v2/vfilter"
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
	ImageFile gi.Filename

	// V1 simple gabor filter parameters
	V1sGabor gabor.Filter

	// geometry of input, output for V1 simple-cell processing
	V1sGeom vfilter.Geom `edit:"-"`

	// neighborhood inhibition for V1s -- each unit gets inhibition from same feature in nearest orthogonal neighbors -- reduces redundancy of feature code
	V1sNeighInhib kwta.NeighInhib

	// kwta parameters for V1s
	V1sKWTA kwta.KWTA

	// target image size to use -- images will be rescaled to this size
	ImgSize image.Point

	// V1 simple gabor filter tensor
	V1sGaborTsr etensor.Float32 `view:"no-inline"`

	// V1 simple gabor filter table (view only)
	V1sGaborTab etable.Table `view:"no-inline"`

	// current input image
	Img image.Image `view:"-"`

	// input image as tensor
	ImgTsr etensor.Float32 `view:"no-inline"`

	// input image reconstructed from V1s tensor
	ImgFromV1sTsr etensor.Float32 `view:"no-inline"`

	// V1 simple gabor filter output tensor
	V1sTsr etensor.Float32 `view:"no-inline"`

	// V1 simple extra Gi from neighbor inhibition tensor
	V1sExtGiTsr etensor.Float32 `view:"no-inline"`

	// V1 simple gabor filter output, kwta output tensor
	V1sKwtaTsr etensor.Float32 `view:"no-inline"`

	// V1 simple gabor filter output, max-pooled 2x2 of V1sKwta tensor
	V1sPoolTsr etensor.Float32 `view:"no-inline"`

	// V1 simple gabor filter output, un-max-pooled 2x2 of V1sPool tensor
	V1sUnPoolTsr etensor.Float32 `view:"no-inline"`

	// V1 simple gabor filter output, angle-only features tensor
	V1sAngOnlyTsr etensor.Float32 `view:"no-inline"`

	// V1 simple gabor filter output, max-pooled 2x2 of AngOnly tensor
	V1sAngPoolTsr etensor.Float32 `view:"no-inline"`

	// V1 complex length sum filter output tensor
	V1cLenSumTsr etensor.Float32 `view:"no-inline"`

	// V1 complex end stop filter output tensor
	V1cEndStopTsr etensor.Float32 `view:"no-inline"`

	// Combined V1 output tensor with V1s simple as first two rows, then length sum, then end stops = 5 rows total
	V1AllTsr etensor.Float32 `view:"no-inline"`

	// inhibition values for V1s KWTA
	V1sInhibs fffb.Inhibs `view:"no-inline"`
}

func (vi *Vis) Defaults() {
	vi.ImageFile = gi.Filename("side-tee-128.png")
	vi.V1sGabor.Defaults()
	sz := 12 // V1mF16 typically = 12, no border
	spc := 4
	vi.V1sGabor.SetSize(sz, spc)
	// note: first arg is border -- we are relying on Geom
	// to set border to .5 * filter size
	// any further border sizes on same image need to add Geom.FiltRt!
	vi.V1sGeom.Set(image.Point{0, 0}, image.Point{spc, spc}, image.Point{sz, sz})
	vi.V1sNeighInhib.Defaults()
	vi.V1sKWTA.Defaults()
	vi.ImgSize = image.Point{128, 128}
	// vi.ImgSize = image.Point{64, 64}
	vi.V1sGabor.ToTensor(&vi.V1sGaborTsr)
	vi.V1sGabor.ToTable(&vi.V1sGaborTab) // note: view only, testing
	vi.V1sGaborTab.Cols[1].SetMetaData("max", "0.05")
	vi.V1sGaborTab.Cols[1].SetMetaData("min", "-0.05")
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
	vfilter.RGBToGrey(vi.Img, &vi.ImgTsr, vi.V1sGeom.FiltRt.X, false) // pad for filt, bot zero
	vfilter.WrapPad(&vi.ImgTsr, vi.V1sGeom.FiltRt.X)
	// vfilter.FadePad(&vi.ImgTsr, vi.V1sGeom.FiltRt.X)
	vi.ImgTsr.SetMetaData("image", "+")
	return nil
}

// V1Simple runs V1Simple Gabor filtering on input image
// must have valid Img in place to start.
// Runs kwta and pool steps after gabor filter.
func (vi *Vis) V1Simple() {
	vfilter.Conv(&vi.V1sGeom, &vi.V1sGaborTsr, &vi.ImgTsr, &vi.V1sTsr, vi.V1sGabor.Gain)
	if vi.V1sNeighInhib.On {
		vi.V1sNeighInhib.Inhib4(&vi.V1sTsr, &vi.V1sExtGiTsr)
	} else {
		vi.V1sExtGiTsr.SetZeros()
	}
	if vi.V1sKWTA.On {
		vi.V1sKWTA.KWTAPool(&vi.V1sTsr, &vi.V1sKwtaTsr, &vi.V1sInhibs, &vi.V1sExtGiTsr)
	} else {
		vi.V1sKwtaTsr.CopyFrom(&vi.V1sTsr)
	}
}

// ImgFromV1Simple reverses V1Simple Gabor filtering from V1s back to input image
func (vi *Vis) ImgFromV1Simple() {
	vi.V1sUnPoolTsr.CopyShapeFrom(&vi.V1sTsr)
	vi.V1sUnPoolTsr.SetZeros()
	vi.ImgFromV1sTsr.CopyShapeFrom(&vi.ImgTsr)
	vi.ImgFromV1sTsr.SetZeros()
	vfilter.UnPool(image.Point{2, 2}, image.Point{2, 2}, &vi.V1sUnPoolTsr, &vi.V1sPoolTsr, true)
	vfilter.Deconv(&vi.V1sGeom, &vi.V1sGaborTsr, &vi.ImgFromV1sTsr, &vi.V1sUnPoolTsr, vi.V1sGabor.Gain)
	norm.Unit32(vi.ImgFromV1sTsr.Values)
	vi.ImgFromV1sTsr.SetMetaData("image", "+")
}

// V1Complex runs V1 complex filters on top of V1Simple features.
// it computes Angle-only, max-pooled version of V1Simple inputs.
func (vi *Vis) V1Complex() {
	vfilter.MaxPool(image.Point{2, 2}, image.Point{2, 2}, &vi.V1sKwtaTsr, &vi.V1sPoolTsr)
	vfilter.MaxReduceFilterY(&vi.V1sKwtaTsr, &vi.V1sAngOnlyTsr)
	vfilter.MaxPool(image.Point{2, 2}, image.Point{2, 2}, &vi.V1sAngOnlyTsr, &vi.V1sAngPoolTsr)
	v1complex.LenSum4(&vi.V1sAngPoolTsr, &vi.V1cLenSumTsr)
	v1complex.EndStop4(&vi.V1sAngPoolTsr, &vi.V1cLenSumTsr, &vi.V1cEndStopTsr)
}

// V1All aggregates all the relevant simple and complex features
// into the V1AllTsr which is used for input to a network
func (vi *Vis) V1All() {
	ny := vi.V1sPoolTsr.Dim(0)
	nx := vi.V1sPoolTsr.Dim(1)
	nang := vi.V1sPoolTsr.Dim(3)
	nrows := 5
	oshp := []int{ny, nx, nrows, nang}
	if !etensor.EqualInts(oshp, vi.V1AllTsr.Shp) {
		vi.V1AllTsr.SetShape(oshp, nil, []string{"Y", "X", "Polarity", "Angle"})
	}
	// 1 length-sum
	vfilter.FeatAgg([]int{0}, 0, &vi.V1cLenSumTsr, &vi.V1AllTsr)
	// 2 end-stop
	vfilter.FeatAgg([]int{0, 1}, 1, &vi.V1cEndStopTsr, &vi.V1AllTsr)
	// 2 pooled simple cell
	vfilter.FeatAgg([]int{0, 1}, 3, &vi.V1sPoolTsr, &vi.V1AllTsr)
}

// Filter is overall method to run filters on current image file name
// loads the image from ImageFile and then runs filters
func (vi *Vis) Filter() error { //gti:add
	err := vi.OpenImage(string(vi.ImageFile))
	if err != nil {
		log.Println(err)
		return err
	}
	vi.V1Simple()
	vi.V1Complex()
	vi.V1All()
	vi.ImgFromV1Simple()
	return nil
}

//////////////////////////////////////////////////////////////////////////////
// 		Gui

func (vi *Vis) ConfigGUI() *gi.Body {
	b := gi.NewBody("v1gabor").SetTitle("V1 Gabor Filtering")

	giv.NewStructView(b, "sv").SetStruct(vi)

	b.AddAppBar(func(tb *gi.Toolbar) {
		giv.NewFuncButton(tb, vi.Filter)
	})

	b.NewWindow().Run().Wait()
	return b
}
