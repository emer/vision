// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

//go:generate core generate -add-types

import (
	"image"
	"log"

	"cogentcore.org/core/core"
	"cogentcore.org/core/iox/imagex"
	"cogentcore.org/core/views"
	"github.com/anthonynsimon/bild/transform"
	"github.com/emer/etable/v2/etable"
	"github.com/emer/etable/v2/etensor"
	_ "github.com/emer/etable/v2/etview" // include to get gui views
	"github.com/emer/etable/v2/norm"
	"github.com/emer/vision/v2/colorspace"
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

// Img manages conversion of a bitmap image into tensor formats for
// subsequent processing by filters.
type V1Img struct { //types:add

	// name of image file to operate on
	File core.Filename

	// target image size to use -- images will be rescaled to this size
	Size image.Point

	// current input image
	Img image.Image `view:"-"`

	// input image as an RGB tensor
	Tsr etensor.Float32 `view:"no-inline"`

	// LMS components + opponents tensor version of image
	LMS etensor.Float32 `view:"no-inline"`
}

func (vi *V1Img) Defaults() {
	vi.Size = image.Point{128, 128}
}

// OpenImage opens given filename as current image Img
// and converts to a float32 tensor for processing
func (vi *V1Img) OpenImage(filepath string, filtsz int) error { //types:add
	var err error
	vi.Img, _, err = imagex.Open(filepath)
	if err != nil {
		log.Println(err)
		return err
	}
	isz := vi.Img.Bounds().Size()
	if isz != vi.Size {
		vi.Img = transform.Resize(vi.Img, vi.Size.X, vi.Size.Y, transform.Linear)
	}
	vfilter.RGBToTensor(vi.Img, &vi.Tsr, filtsz, false) // pad for filt, bot zero
	vfilter.WrapPadRGB(&vi.Tsr, filtsz)
	colorspace.RGBTensorToLMSComps(&vi.LMS, &vi.Tsr)
	vi.Tsr.SetMetaData("image", "+")
	vi.Tsr.SetMetaData("min", "0")
	return nil
}

// V1sOut contains output tensors for V1 Simple filtering, one per opponnent
type V1sOut struct { //types:add

	// V1 simple gabor filter output tensor
	Tsr etensor.Float32 `view:"no-inline"`

	// V1 simple extra Gi from neighbor inhibition tensor
	ExtGiTsr etensor.Float32 `view:"no-inline"`

	// V1 simple gabor filter output, kwta output tensor
	KwtaTsr etensor.Float32 `view:"no-inline"`

	// V1 simple gabor filter output, max-pooled 2x2 of Kwta tensor
	PoolTsr etensor.Float32 `view:"no-inline"`
}

// Vis encapsulates specific visual processing pipeline in
// use in a given case -- can add / modify this as needed.
// Handles 3 major opponent channels: WhiteBlack, RedGreen, BlueYellow
type Vis struct { //types:add

	// if true, do full color filtering -- else Black/White only
	Color bool

	// record separate rows in V1s summary for each color -- otherwise just records the max across all colors
	SepColor bool

	// extra gain for color channels -- lower contrast in general
	ColorGain float32 `default:"8"`

	// image that we operate upon -- one image often shared among multiple filters
	Img *V1Img

	// V1 simple gabor filter parameters
	V1sGabor gabor.Filter

	// geometry of input, output for V1 simple-cell processing
	V1sGeom vfilter.Geom `edit:"-"`

	// neighborhood inhibition for V1s -- each unit gets inhibition from same feature in nearest orthogonal neighbors -- reduces redundancy of feature code
	V1sNeighInhib kwta.NeighInhib

	// kwta parameters for V1s
	V1sKWTA kwta.KWTA

	// V1 simple gabor filter tensor
	V1sGaborTsr etensor.Float32 `view:"no-inline"`

	// V1 simple gabor filter table (view only)
	V1sGaborTab etable.Table `view:"no-inline"`

	// V1 simple gabor filter output, per channel
	V1s [colorspace.OpponentsN]V1sOut `view:"inline"`

	// max over V1 simple gabor filters output tensor
	V1sMaxTsr etensor.Float32 `view:"no-inline"`

	// V1 simple gabor filter output, max-pooled 2x2 of Kwta tensor
	V1sPoolTsr etensor.Float32 `view:"no-inline"`

	// V1 simple gabor filter output, un-max-pooled 2x2 of Pool tensor
	V1sUnPoolTsr etensor.Float32 `view:"no-inline"`

	// input image reconstructed from V1s tensor
	ImgFromV1sTsr etensor.Float32 `view:"no-inline"`

	// V1 simple gabor filter output, angle-only features tensor
	V1sAngOnlyTsr etensor.Float32 `view:"no-inline"`

	// V1 simple gabor filter output, max-pooled 2x2 of AngOnly tensor
	V1sAngPoolTsr etensor.Float32 `view:"no-inline"`

	// V1 complex length sum filter output tensor
	V1cLenSumTsr etensor.Float32 `view:"no-inline"`

	// V1 complex end stop filter output tensor
	V1cEndStopTsr etensor.Float32 `view:"no-inline"`

	// Combined V1 output tensor with V1s simple as first two rows, then length sum, then end stops = 5 rows total (9 if SepColor)
	V1AllTsr etensor.Float32 `view:"no-inline"`

	// inhibition values for V1s KWTA
	V1sInhibs fffb.Inhibs `view:"no-inline"`
}

func (vi *Vis) Defaults() {
	vi.Color = true
	vi.SepColor = true
	vi.ColorGain = 8
	vi.Img = &V1Img{}
	vi.Img.Defaults()
	vi.Img.File = core.Filename("car_004_00001.png")
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
	// vi.ImgSize = image.Point{64, 64}
	vi.V1sGabor.ToTensor(&vi.V1sGaborTsr)
	vi.V1sGabor.ToTable(&vi.V1sGaborTab) // note: view only, testing
	vi.V1sGaborTab.Cols[1].SetMetaData("max", "0.05")
	vi.V1sGaborTab.Cols[1].SetMetaData("min", "-0.05")
}

// V1SimpleImg runs V1Simple Gabor filtering on input image
// Runs kwta and pool steps after gabor filter.
// has extra gain factor -- > 1 for color contrasts.
func (vi *Vis) V1SimpleImg(v1s *V1sOut, img *etensor.Float32, gain float32) {
	vfilter.Conv(&vi.V1sGeom, &vi.V1sGaborTsr, img, &v1s.Tsr, gain*vi.V1sGabor.Gain)
	if vi.V1sNeighInhib.On {
		vi.V1sNeighInhib.Inhib4(&v1s.Tsr, &v1s.ExtGiTsr)
	} else {
		v1s.ExtGiTsr.SetZeros()
	}
	if vi.V1sKWTA.On {
		vi.V1sKWTA.KWTAPool(&v1s.Tsr, &v1s.KwtaTsr, &vi.V1sInhibs, &v1s.ExtGiTsr)
	} else {
		v1s.KwtaTsr.CopyFrom(&v1s.Tsr)
	}
}

// V1Simple runs all V1Simple Gabor filtering, depending on Color
func (vi *Vis) V1Simple() {
	grey := vi.Img.LMS.SubSpace([]int{int(colorspace.GREY)}).(*etensor.Float32)
	wbout := &vi.V1s[colorspace.WhiteBlack]
	vi.V1SimpleImg(wbout, grey, 1)
	vi.V1sMaxTsr.CopyShapeFrom(&wbout.KwtaTsr)
	vi.V1sMaxTsr.CopyFrom(&wbout.KwtaTsr)
	if vi.Color {
		rgout := &vi.V1s[colorspace.RedGreen]
		rgimg := vi.Img.LMS.SubSpace([]int{int(colorspace.LvMC)}).(*etensor.Float32)
		vi.V1SimpleImg(rgout, rgimg, vi.ColorGain)
		byout := &vi.V1s[colorspace.BlueYellow]
		byimg := vi.Img.LMS.SubSpace([]int{int(colorspace.SvLMC)}).(*etensor.Float32)
		vi.V1SimpleImg(byout, byimg, vi.ColorGain)
		for i, vl := range vi.V1sMaxTsr.Values {
			rg := rgout.KwtaTsr.Values[i]
			by := byout.KwtaTsr.Values[i]
			if rg > vl {
				vl = rg
			}
			if by > vl {
				vl = by
			}
			vi.V1sMaxTsr.Values[i] = vl
		}
	}
}

// ImgFromV1Simple reverses V1Simple Gabor filtering from V1s back to input image
func (vi *Vis) ImgFromV1Simple() {
	vi.V1sUnPoolTsr.CopyShapeFrom(&vi.V1sMaxTsr)
	vi.V1sUnPoolTsr.SetZeros()
	vi.ImgFromV1sTsr.SetShape(vi.Img.Tsr.Shapes()[1:], nil, []string{"Y", "X"})
	vi.ImgFromV1sTsr.SetZeros()
	vfilter.UnPool(image.Point{2, 2}, image.Point{2, 2}, &vi.V1sUnPoolTsr, &vi.V1sPoolTsr, true)
	vfilter.Deconv(&vi.V1sGeom, &vi.V1sGaborTsr, &vi.ImgFromV1sTsr, &vi.V1sUnPoolTsr, vi.V1sGabor.Gain)
	norm.Unit32(vi.ImgFromV1sTsr.Values)
	vi.ImgFromV1sTsr.SetMetaData("image", "+")
}

// V1Complex runs V1 complex filters on top of V1Simple features.
// it computes Angle-only, max-pooled version of V1Simple inputs.
func (vi *Vis) V1Complex() {
	vfilter.MaxPool(image.Point{2, 2}, image.Point{2, 2}, &vi.V1sMaxTsr, &vi.V1sPoolTsr)
	vfilter.MaxReduceFilterY(&vi.V1sMaxTsr, &vi.V1sAngOnlyTsr)
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
	if vi.Color && vi.SepColor {
		nrows += 4
	}
	oshp := []int{ny, nx, nrows, nang}
	if !etensor.EqualInts(oshp, vi.V1AllTsr.Shp) {
		vi.V1AllTsr.SetShape(oshp, nil, []string{"Y", "X", "Polarity", "Angle"})
	}
	// 1 length-sum
	vfilter.FeatAgg([]int{0}, 0, &vi.V1cLenSumTsr, &vi.V1AllTsr)
	// 2 end-stop
	vfilter.FeatAgg([]int{0, 1}, 1, &vi.V1cEndStopTsr, &vi.V1AllTsr)
	// 2 pooled simple cell
	if vi.Color && vi.SepColor {
		rgout := &vi.V1s[colorspace.RedGreen]
		byout := &vi.V1s[colorspace.BlueYellow]
		vfilter.MaxPool(image.Point{2, 2}, image.Point{2, 2}, &rgout.KwtaTsr, &rgout.PoolTsr)
		vfilter.MaxPool(image.Point{2, 2}, image.Point{2, 2}, &byout.KwtaTsr, &byout.PoolTsr)
		vfilter.FeatAgg([]int{0, 1}, 5, &rgout.PoolTsr, &vi.V1AllTsr)
		vfilter.FeatAgg([]int{0, 1}, 7, &byout.PoolTsr, &vi.V1AllTsr)
	} else {
		vfilter.FeatAgg([]int{0, 1}, 3, &vi.V1sPoolTsr, &vi.V1AllTsr)
	}
}

// Filter is overall method to run filters on current image file name
// loads the image from ImageFile and then runs filters
func (vi *Vis) Filter() error { //types:add
	err := vi.Img.OpenImage(string(vi.Img.File), vi.V1sGeom.FiltRt.X)
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

func (vi *Vis) ConfigGUI() *core.Body {
	b := core.NewBody("color-gabor").SetTitle("V1 Color Gabor Filtering")

	views.NewStructView(b, "sv").SetStruct(vi)

	b.AddAppBar(func(tb *core.Toolbar) {
		views.NewFuncButton(tb, vi.Filter)
	})

	b.RunMainWindow()
	return b
}
