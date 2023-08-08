// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"image"
	"log"

	"github.com/anthonynsimon/bild/transform"
	"github.com/emer/etable/etable"
	"github.com/emer/etable/etensor"
	_ "github.com/emer/etable/etview" // include to get gui views
	"github.com/emer/etable/norm"
	"github.com/emer/leabra/fffb"
	"github.com/emer/vision/colorspace"
	"github.com/emer/vision/gabor"
	"github.com/emer/vision/kwta"
	"github.com/emer/vision/v1complex"
	"github.com/emer/vision/vfilter"
	"github.com/goki/gi/gi"
	"github.com/goki/gi/gimain"
	"github.com/goki/gi/giv"
	"github.com/goki/ki/ki"
	"github.com/goki/ki/kit"
	"github.com/goki/mat32"
)

// this is the stub main for gogi that calls our actual
// mainrun function, at end of file
func main() {
	gimain.Main(func() {
		mainrun()
	})
}

// Img manages conversion of a bitmap image into tensor formats for
// subsequent processing by filters.
type V1Img struct {

	// name of image file to operate on
	File gi.FileName `desc:"name of image file to operate on"`

	// target image size to use -- images will be rescaled to this size
	Size image.Point `desc:"target image size to use -- images will be rescaled to this size"`

	// [view: -] current input image
	Img image.Image `view:"-" desc:"current input image"`

	// [view: no-inline] input image as an RGB tensor
	Tsr etensor.Float32 `view:"no-inline" desc:"input image as an RGB tensor"`

	// [view: no-inline] LMS components + opponents tensor version of image
	LMS etensor.Float32 `view:"no-inline" desc:"LMS components + opponents tensor version of image"`
}

func (vi *V1Img) Defaults() {
	vi.Size = image.Point{128, 128}
}

// OpenImage opens given filename as current image Img
// and converts to a float32 tensor for processing
func (vi *V1Img) OpenImage(filepath string, filtsz int) error {
	var err error
	vi.Img, err = gi.OpenImage(filepath)
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
type V1sOut struct {

	// [view: no-inline] V1 simple gabor filter output tensor
	Tsr etensor.Float32 `view:"no-inline" desc:"V1 simple gabor filter output tensor"`

	// [view: no-inline] V1 simple extra Gi from neighbor inhibition tensor
	ExtGiTsr etensor.Float32 `view:"no-inline" desc:"V1 simple extra Gi from neighbor inhibition tensor"`

	// [view: no-inline] V1 simple gabor filter output, kwta output tensor
	KwtaTsr etensor.Float32 `view:"no-inline" desc:"V1 simple gabor filter output, kwta output tensor"`

	// [view: no-inline] V1 simple gabor filter output, max-pooled 2x2 of Kwta tensor
	PoolTsr etensor.Float32 `view:"no-inline" desc:"V1 simple gabor filter output, max-pooled 2x2 of Kwta tensor"`
}

// Vis encapsulates specific visual processing pipeline in
// use in a given case -- can add / modify this as needed.
// Handles 3 major opponent channels: WhiteBlack, RedGreen, BlueYellow
type Vis struct {

	// if true, do full color filtering -- else Black/White only
	Color bool `desc:"if true, do full color filtering -- else Black/White only"`

	// record separate rows in V1s summary for each color -- otherwise just records the max across all colors
	SepColor bool `desc:"record separate rows in V1s summary for each color -- otherwise just records the max across all colors"`

	// [def: 8] extra gain for color channels -- lower contrast in general
	ColorGain float32 `def:"8" desc:"extra gain for color channels -- lower contrast in general"`

	// image that we operate upon -- one image often shared among multiple filters
	Img *V1Img `desc:"image that we operate upon -- one image often shared among multiple filters"`

	// V1 simple gabor filter parameters
	V1sGabor gabor.Filter `desc:"V1 simple gabor filter parameters"`

	// [view: inline] geometry of input, output for V1 simple-cell processing
	V1sGeom vfilter.Geom `inactive:"+" view:"inline" desc:"geometry of input, output for V1 simple-cell processing"`

	// neighborhood inhibition for V1s -- each unit gets inhibition from same feature in nearest orthogonal neighbors -- reduces redundancy of feature code
	V1sNeighInhib kwta.NeighInhib `desc:"neighborhood inhibition for V1s -- each unit gets inhibition from same feature in nearest orthogonal neighbors -- reduces redundancy of feature code"`

	// kwta parameters for V1s
	V1sKWTA kwta.KWTA `desc:"kwta parameters for V1s"`

	// [view: no-inline] V1 simple gabor filter tensor
	V1sGaborTsr etensor.Float32 `view:"no-inline" desc:"V1 simple gabor filter tensor"`

	// [view: no-inline] V1 simple gabor filter table (view only)
	V1sGaborTab etable.Table `view:"no-inline" desc:"V1 simple gabor filter table (view only)"`

	// [view: inline] V1 simple gabor filter output, per channel
	V1s [colorspace.OpponentsN]V1sOut `view:"inline" desc:"V1 simple gabor filter output, per channel"`

	// [view: no-inline] max over V1 simple gabor filters output tensor
	V1sMaxTsr etensor.Float32 `view:"no-inline" desc:"max over V1 simple gabor filters output tensor"`

	// [view: no-inline] V1 simple gabor filter output, max-pooled 2x2 of Kwta tensor
	V1sPoolTsr etensor.Float32 `view:"no-inline" desc:"V1 simple gabor filter output, max-pooled 2x2 of Kwta tensor"`

	// [view: no-inline] V1 simple gabor filter output, un-max-pooled 2x2 of Pool tensor
	V1sUnPoolTsr etensor.Float32 `view:"no-inline" desc:"V1 simple gabor filter output, un-max-pooled 2x2 of Pool tensor"`

	// [view: no-inline] input image reconstructed from V1s tensor
	ImgFmV1sTsr etensor.Float32 `view:"no-inline" desc:"input image reconstructed from V1s tensor"`

	// [view: no-inline] V1 simple gabor filter output, angle-only features tensor
	V1sAngOnlyTsr etensor.Float32 `view:"no-inline" desc:"V1 simple gabor filter output, angle-only features tensor"`

	// [view: no-inline] V1 simple gabor filter output, max-pooled 2x2 of AngOnly tensor
	V1sAngPoolTsr etensor.Float32 `view:"no-inline" desc:"V1 simple gabor filter output, max-pooled 2x2 of AngOnly tensor"`

	// [view: no-inline] V1 complex length sum filter output tensor
	V1cLenSumTsr etensor.Float32 `view:"no-inline" desc:"V1 complex length sum filter output tensor"`

	// [view: no-inline] V1 complex end stop filter output tensor
	V1cEndStopTsr etensor.Float32 `view:"no-inline" desc:"V1 complex end stop filter output tensor"`

	// [view: no-inline] Combined V1 output tensor with V1s simple as first two rows, then length sum, then end stops = 5 rows total (9 if SepColor)
	V1AllTsr etensor.Float32 `view:"no-inline" desc:"Combined V1 output tensor with V1s simple as first two rows, then length sum, then end stops = 5 rows total (9 if SepColor)"`

	// [view: no-inline] inhibition values for V1s KWTA
	V1sInhibs fffb.Inhibs `view:"no-inline" desc:"inhibition values for V1s KWTA"`
}

var KiT_Vis = kit.Types.AddType(&Vis{}, VisProps)

func (vi *Vis) Defaults() {
	vi.Color = true
	vi.SepColor = true
	vi.ColorGain = 8
	vi.Img = &V1Img{}
	vi.Img.Defaults()
	vi.Img.File = gi.FileName("car_004_00001.png")
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

// ImgFmV1Simple reverses V1Simple Gabor filtering from V1s back to input image
func (vi *Vis) ImgFmV1Simple() {
	vi.V1sUnPoolTsr.CopyShapeFrom(&vi.V1sMaxTsr)
	vi.V1sUnPoolTsr.SetZeros()
	vi.ImgFmV1sTsr.SetShape(vi.Img.Tsr.Shapes()[1:], nil, []string{"Y", "X"})
	vi.ImgFmV1sTsr.SetZeros()
	vfilter.UnPool(image.Point{2, 2}, image.Point{2, 2}, &vi.V1sUnPoolTsr, &vi.V1sPoolTsr, true)
	vfilter.Deconv(&vi.V1sGeom, &vi.V1sGaborTsr, &vi.ImgFmV1sTsr, &vi.V1sUnPoolTsr, vi.V1sGabor.Gain)
	norm.Unit32(vi.ImgFmV1sTsr.Values)
	vi.ImgFmV1sTsr.SetMetaData("image", "+")
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
func (vi *Vis) Filter() error {
	err := vi.Img.OpenImage(string(vi.Img.File), vi.V1sGeom.FiltRt.X)
	if err != nil {
		log.Println(err)
		return err
	}
	vi.V1Simple()
	vi.V1Complex()
	vi.V1All()
	vi.ImgFmV1Simple()
	return nil
}

////////////////////////////////////////////////////////////////////////////////////////////
// 		Gui

// ConfigGui configures the GoGi gui interface for this Vis
func (vi *Vis) ConfigGui() *gi.Window {
	width := 1600
	height := 1200

	gi.SetAppName("color_gabor")
	gi.SetAppAbout(`This demonstrates color-sensitive V1 Gabor Filtering.  See <a href="https://github.com/emer/vision">Vision on GitHub</a>.</p>`)

	win := gi.NewMainWindow("color_gabor", "V1 Color Gabor Filtering", width, height)
	// vi.Win = win

	vp := win.WinViewport2D()
	updt := vp.UpdateStart()

	mfr := win.SetMainFrame()

	tbar := gi.AddNewToolBar(mfr, "tbar")
	tbar.SetStretchMaxWidth()
	// vi.ToolBar = tbar

	split := gi.AddNewSplitView(mfr, "split")
	split.Dim = mat32.X
	split.SetStretchMax()

	sv := giv.AddNewStructView(split, "sv")
	sv.Viewport = vp
	sv.SetStruct(vi)

	split.SetSplits(1)

	// main menu
	appnm := gi.AppName()
	mmen := win.MainMenu
	mmen.ConfigMenus([]string{appnm, "File", "Edit", "Window"})

	amen := win.MainMenu.ChildByName(appnm, 0).(*gi.Action)
	amen.Menu.AddAppMenu(win)

	emen := win.MainMenu.ChildByName("Edit", 1).(*gi.Action)
	emen.Menu.AddCopyCutPaste(win)

	gi.SetQuitReqFunc(func() {
		gi.Quit()
	})
	win.SetCloseReqFunc(func(w *gi.Window) {
		gi.Quit()
	})
	win.SetCloseCleanFunc(func(w *gi.Window) {
		go gi.Quit() // once main window is closed, quit
	})

	vp.UpdateEndNoSig(updt)

	win.MainMenuUpdated()
	return win
}

// These props create interactive toolbar for GUI
var VisProps = ki.Props{
	"ToolBar": ki.PropSlice{
		{"Filter", ki.Props{
			"desc": "run filter methods on current ImageFile image",
			"icon": "updt",
		}},
	},
}

var TheVis Vis

func mainrun() {
	TheVis.Defaults()
	TheVis.Filter()
	win := TheVis.ConfigGui()
	win.StartEventLoop()
}
